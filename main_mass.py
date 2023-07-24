import torch
import time
import os
from tqdm import tqdm
import numpy as np
import sys
import copy
import igl

sys.path.append(os.path.join(os.path.dirname(__file__), "ext/large-steps"))
from largesteps.optimize import AdamUniform
from largesteps.geometry import compute_matrix, laplacian_uniform, laplacian_cot
from largesteps.parameterize import to_differential, from_differential
from scripts.render import NVDRenderer
from scripts.load_xml import load_scene
from scripts.constants import REMESH_DIR
from scripts.geometry import remove_duplicates, compute_face_normals, compute_vertex_normals, average_edge_length, massmatrix_voronoi
sys.path.append(REMESH_DIR)
# from pyremesh import remesh_botsch

def full_area(verts, faces, eps=1e-18):
    face_verts = verts[faces]
    barycentric = face_verts.mean(-2).unsqueeze(-2)
    fv_vec = face_verts - barycentric
        
    area = 0.5 * (fv_vec[:,0].cross(fv_vec[:,1]) + fv_vec[:,1].cross(fv_vec[:,2]) + fv_vec[:,2].cross(fv_vec[:,0])).norm(dim=1).abs().clamp_(min=eps)
    idx = faces.view(-1)
    v_areas = torch.zeros(verts.shape[0], dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    v_areas.scatter_add_(0, idx, val)
    return v_areas

def vert_area(verts, faces, eps=1e-18):
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)
    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=eps).sqrt()
    idx = faces.view(-1)
    v_areas = torch.zeros(verts.shape[0], dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    v_areas.scatter_add_(0, idx, val)
    return v_areas

def massmatrix_voronoi_approx(verts, faces, eps=1e-12):
    """
    Compute the area of the Voronoi cell around each vertex in the mesh.
    https://mathworld.wolfram.com/BarycentricCoordinates.html
    """
    l0 = (verts[faces[:,1]] - verts[faces[:,2]]).norm(dim=1)#.clamp_(min=eps)
    l1 = (verts[faces[:,2]] - verts[faces[:,0]]).norm(dim=1)#.clamp_(min=eps)
    l2 = (verts[faces[:,0]] - verts[faces[:,1]]).norm(dim=1)#.clamp_(min=eps)
    l = torch.stack((l0, l1, l2), dim=1)
    return torch.zeros_like(verts).scatter_add_(0, faces, l, ).mean(dim=1)

def cots(verts, faces):
    """
    Compute the cotangent laplacian
    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    # V = sum(V_n), F = sum(F_n)
    V, F = verts.shape[0], faces.shape[0]
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)
    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()
    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0
    return cot

# areaarea = massmatrix_voronoi
areaarea = massmatrix_voronoi_approx
# areaarea = full_area

def mass_loss(V, F, L, mat, m_ori_mean, boundaries = []):
    tmp_Mass = areaarea(V, F)
    # tmp_Mass = massmatrix_voronoi_approx(V, F)
    # tmp_Mass = full_area(V, F)
    with torch.no_grad():
        m_mean = tmp_Mass.mean()
        # m_mean = tmp_Mass.median()
    with torch.no_grad():
        # lap_c = (L @ V) / m_ori_mean
        lap_c = (L @ V) / m_mean
        if len(boundaries) > 0:
            lap_c[boundaries] *= 1e-12# lap_c.min()
        density = torch.linalg.norm(lap_c, dim=-1).unsqueeze(-1)
        density = from_differential(mat, density)
        density = from_differential(mat, density)
        density = from_differential(mat, density)
        density = density / density.mean()
        density = torch.reciprocal(density)
        density = torch.clamp(density, 0.0, 1.0)
        MM_d = tmp_Mass * density.squeeze()
        # MM_d = torch.clamp(MM_d, min = m_ori_mean / 4)
    mass_mean_loss = (tmp_Mass - m_mean).square().mean() / m_mean
    # mass_mean_loss = (m_mean - tmp_Mass).clamp_(min=0).square().mean()
    # mass_mean_loss = (L @ tmp_Mass.T).square().mean()
    mass_lap_loss =  (tmp_Mass - MM_d).square().mean() / m_mean

    return mass_mean_loss, mass_lap_loss



def optimize_shape(filepath, params):
    """
    Optimize a shape given a scene.

    This will expect a Mitsuba scene as input containing the cameras, envmap and
    source and target models.

    Parameters
    ----------
    filepath : str Path to the XML file of the scene to optimize. params : dict
        Dictionary containing all optimization parameters.
    """
    opt_time = params.get("time", -1) # Optimization time (in minutes)
    steps = params.get("steps", 100) # Number of optimization steps (ignored if time > 0)
    step_size = params.get("step_size", 0.01) # Step size
    boost = params.get("boost", 1) # Gradient boost used in nvdiffrast
    smooth = params.get("smooth", True) # Use our method or not
    shading = params.get("shading", True) # Use shading, otherwise render silhouettes
    reg = params.get("reg", 0.0) # Regularization weight
    solver = params.get("solver", 'Cholesky') # Solver to use
    lambda_ = params.get("lambda", 1.0) # Hyperparameter lambda of our method, used to compute the parameterization matrix as (I + lambda_ * L)
    alpha = params.get("alpha", None) # Alternative hyperparameter, used to compute the parameterization matrix as ((1-alpha) * I + alpha * L)
    remesh = params.get("remesh", -1) # Time step(s) at which to remesh
    optimizer = params.get("optimizer", AdamUniform) # Which optimizer to use
    use_tr = params.get("use_tr", True) # Optimize a global translation at the same time
    loss_function = params.get("loss", "l2") # Which loss to use
    bilaplacian = params.get("bilaplacian", True) # Use the bilaplacian or the laplacian regularization loss
    mass_weight = params.get("mass_weights", 1.8) # Use the bilaplacian or the laplacian regularization loss

    # Load the scene
    scene_params = load_scene(filepath)

    # Load reference shape
    v_ref = scene_params["mesh-target"]["vertices"]
    f_ref = scene_params["mesh-target"]["faces"]
    if "normals" in scene_params["mesh-target"].keys():
        n_ref = scene_params["mesh-target"]["normals"]
    else:
        face_normals = compute_face_normals(v_ref, f_ref)
        n_ref = compute_vertex_normals(v_ref, f_ref, face_normals)

    # Load source shape
    v_src = scene_params["mesh-source"]["vertices"]
    f_src = scene_params["mesh-source"]["faces"]
    # Remove duplicates. This is necessary to avoid seams of meshes to rip apart during the optimization
    v_unique, f_unique, duplicate_idx = remove_duplicates(v_src, f_src)

    # Initialize the renderer
    renderer = NVDRenderer(scene_params, shading=shading, boost=boost)

    # Render the reference images
    ref_imgs = renderer.render(v_ref, n_ref, f_ref)

    # Compute the laplacian for the regularization term
    L = laplacian_uniform(v_unique, f_unique)
    ori_C = cots(v_unique, f_unique)
    # mass_ori_mean = full_area(v_unique, f_unique).mean()
    mass_ori_mean = areaarea(v_unique, f_unique).mean()
    # mass_ori_mean = massmatrix_voronoi(v_unique, f_unique).mean()
    boundaries = igl.boundary_loop(f_unique.cpu().numpy())
    
    # Initialize the optimized variables and the optimizer
    tr = torch.zeros((1,3), device='cuda', dtype=torch.float32)

    if smooth:
        # Compute the system matrix and parameterize
        M = compute_matrix(v_unique, f_unique, lambda_=lambda_, alpha=alpha)
        # M_mass = compute_matrix(v_unique, f_unique, lambda_=100)
        M_mass = compute_matrix(v_unique, f_unique, None, alpha=0.5)
        u_unique = to_differential(M, v_unique)

    def initialize_optimizer(u, v, tr, step_size):
        """
        Initialize the optimizer

        Parameters
        ----------
        - u : torch.Tensor or None
            Parameterized coordinates to optimize if not None
        - v : torch.Tensor
            Cartesian coordinates to optimize if u is None
        - tr : torch.Tensor
            Global translation to optimize if not None
        - step_size : float
            Step size

        Returns
        -------
        a torch.optim.Optimizer containing the tensors to optimize.
        """
        opt_params = []
        if tr is not None:
            tr.requires_grad = True
            opt_params.append(tr)
        if u is not None:
            u.requires_grad = True
            opt_params.append(u)
        else:
            v.requires_grad = True
            opt_params.append(v)

        return optimizer(opt_params, lr=step_size)

    opt = initialize_optimizer(u_unique if smooth else None, v_unique, tr if use_tr else None, step_size)

    # Set values for time and step count
    if opt_time > 0:
        steps = -1
    it = 0
    t0 = time.perf_counter()
    t = t0
    opt_time *= 60

    # Dictionary that is returned in the end, contains useful information for debug/analysis
    result_dict = {"vert_steps": [], "tr_steps": [], "f": [f_src.cpu().numpy().copy()],
                "losses": [], "im_ref": ref_imgs.cpu().numpy().copy(), "im":[],
                "v_ref": v_ref.cpu().numpy().copy(), "f_ref": f_ref.cpu().numpy().copy()}

    if type(remesh) == list:
        remesh_it = remesh.pop(0)
    else:
        remesh_it = remesh


    # Optimization loop
    with tqdm(total=max(steps, opt_time), ncols=100, bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
        while it < steps or (t-t0) < opt_time:
            if it == remesh_it:
                # Remesh
                with torch.no_grad():
                    if smooth:
                        v_unique = from_differential(M, u_unique, solver)

                    v_cpu = v_unique.cpu().numpy()
                    f_cpu = f_unique.cpu().numpy()
                    # Target edge length
                    h = (average_edge_length(v_unique, f_unique)).cpu().numpy()*0.5

                    # Run 5 iterations of the Botsch-Kobbelt remeshing algorithm
                    v_new, f_new = remesh_botsch(v_cpu.astype(np.double), f_cpu.astype(np.int32), 5, h, True)

                    v_src = torch.from_numpy(v_new).cuda().float().contiguous()
                    f_src = torch.from_numpy(f_new).cuda().contiguous()

                    v_unique, f_unique, duplicate_idx = remove_duplicates(v_src, f_src)
                    result_dict["f"].append(f_new)
                    # Recompute laplacian
                    L = laplacian_uniform(v_unique, f_unique)

                    if smooth:
                        # Compute the system matrix and parameterize
                        M = compute_matrix(v_unique, f_unique, lambda_=lambda_, alpha=alpha)
                        u_unique = to_differential(M, v_unique)

                    step_size *= 0.8
                    opt = initialize_optimizer(u_unique if smooth else None, v_unique, tr if use_tr else None, step_size)

                # Get next remesh iteration if any
                if type(remesh) == list and len(remesh) > 0:
                    remesh_it = remesh.pop(0)

            # Get cartesian coordinates
            if smooth:
                v_unique = from_differential(M, u_unique, solver)

            # Get the version of the mesh with the duplicates
            v_opt = v_unique[duplicate_idx]
            # Recompute vertex normals
            face_normals = compute_face_normals(v_unique, f_unique)
            n_unique = compute_vertex_normals(v_unique, f_unique, face_normals)
            n_opt = n_unique[duplicate_idx]

            # Render images
            opt_imgs = renderer.render(tr + v_opt, n_opt, f_src)

            # Compute image loss
            if loss_function == "l1":
                im_loss = (opt_imgs - ref_imgs).abs().mean()
            elif loss_function == "l2":
                im_loss = (opt_imgs - ref_imgs).square().mean()

            # Add regularization
            if bilaplacian:
                reg_loss = (L@v_unique).square().mean()
            else:
                reg_loss = (v_unique * (L @v_unique)).mean()
                
            loss = im_loss + reg * reg_loss
            
            mass_mean_loss, mass_lap_loss = mass_loss(v_unique, f_unique, L, M_mass, mass_ori_mean, boundaries)
           
            
            if it < steps/4:
                loss += mass_weight * (mass_mean_loss)
            elif it < steps/2:
                # loss += mass_weight * (mass_mean_loss)
                loss += 2 * mass_weight * (mass_lap_loss)
                # for bunny
                if False:
                    loss += 4 * mass_weight * (mass_mean_loss)

            # Record optimization state for later processing
            result_dict["losses"].append((im_loss.detach().cpu().numpy().copy(), (L@v_unique.detach()).square().mean().cpu().numpy().copy()))
            result_dict["vert_steps"].append(v_opt.detach().cpu().numpy().copy())
            result_dict["tr_steps"].append(tr.detach().cpu().numpy().copy())

            # Backpropagate
            opt.zero_grad()
            loss.backward()
            # Update parameters
            opt.step()

            it += 1
            t = time.perf_counter()
            if steps > -1:
                pbar.update(1)
            else:
                pbar.update(min(opt_time, (t-t0)) - pbar.n)

    result_dict["losses"] = np.array(result_dict["losses"])
    return result_dict
