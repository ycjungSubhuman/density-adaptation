import os
import sys
sys.path.append("../")
sys.path.append("./")

import glob
import numpy as np

import igl
import open3d as o3d
import pandas as pd

import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_gather, knn_points

sys.path.append(os.path.join(os.path.dirname(__file__), "ext/large-steps"))
from scripts.geometry import massmatrix_voronoi, compute_vertex_normals
from largesteps.geometry import laplacian_uniform, compute_matrix
from largesteps.parameterize import to_differential, from_differential
from largesteps.optimize import AdamUniform
from pytorch3d.ops import sample_points_from_meshes

import copy

device = torch.device('cuda')
dtype = torch.float32

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

def massmatrix_voronoi_approx(verts, faces):
    """
    Compute the area of the Voronoi cell around each vertex in the mesh.
    https://mathworld.wolfram.com/BarycentricCoordinates.html
    """
    l0 = (verts[faces[:,1]] - verts[faces[:,2]]).norm(dim=1)
    l1 = (verts[faces[:,2]] - verts[faces[:,0]]).norm(dim=1)
    l2 = (verts[faces[:,0]] - verts[faces[:,1]]).norm(dim=1)
    l = torch.stack((l0, l1, l2), dim=1)
    return torch.zeros_like(verts).scatter_add_(0, faces, l, ).mean(dim=1)

areaarea = massmatrix_voronoi_approx


def mass_loss(V, F, L, mat):
    tmp_Mass = areaarea(V, F)
    with torch.no_grad():
        m_mean = tmp_Mass.mean()
    with torch.no_grad():
        lap_c = (L @ V) / m_mean
        
        density = torch.linalg.norm(lap_c, dim=-1).unsqueeze(-1)
        density = from_differential(mat, density)
        density = from_differential(mat, density)
        density = from_differential(mat, density)
        
        density = density / density.mean()
        density = torch.reciprocal(density)
        density = torch.clamp(density, 0.0, 1.0)
        MM_d = tmp_Mass * density.squeeze()
    mass_mean_loss = (tmp_Mass - m_mean).square().mean() / m_mean
    mass_lap_loss =  (tmp_Mass - MM_d).square().mean() / m_mean

    return mass_mean_loss, mass_lap_loss

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", default=0)
args = parser.parse_args()


if __name__ == '__main__':
    tar_dir = "./3dcaricshop/original_data/processedData/rawMesh"
    res_dir = "./3dcaricshop/sphere_fit"
    data_dir = "./3dcaricshop/data"

    tar_files = sorted(glob.glob(os.path.join(tar_dir, "*", "*.obj")))

    # template_density = "2k"
    template_density = "10k"
    V_src_ori, F_src = igl.read_triangle_mesh(os.path.join(data_dir, "sphere_" + template_density + ".ply"))
    n_iter = 800
    mass_weight = 1.5

    # tar_models = [0]
    # tar_models = range(args.start_idx, min(args.start_idx + 50, len(tar_files)))
    tar_models = range(len(tar_files))
    methods = ["large_steps", "ours"]

    for tar_idx in tar_models:
        res_dir_model = os.path.join(res_dir, tar_files[tar_idx].split("/")[-2])
        os.makedirs(res_dir_model, exist_ok=True)
        
        V_tar, F_tar = igl.read_triangle_mesh(tar_files[tar_idx])
        N_tar = np.nan_to_num(igl.per_vertex_normals(V_tar, F_tar))
        n_tar_verts = len(V_tar)

        # Normalize
        V_src = copy.deepcopy(V_src_ori)
        trans_ori = np.mean(V_tar, axis=0)
        V_tar -= trans_ori[np.newaxis, :]
        V_src -= trans_ori[np.newaxis, :]
        scale_ori = np.max(np.linalg.norm(V_tar, axis=-1))
        V_tar /= scale_ori
        V_src /= scale_ori
        
        losses = np.zeros((len(methods) * 2, n_iter))
        table_label = ["Large_steps_cham", "Ours_cham", "Large_steps_norm", "Ours_norm"]
        err_res_path = os.path.join(res_dir_model, os.path.basename(tar_files[tar_idx]).rsplit('.', 1)[0] + "_{}.csv".format(template_density))
        
        for k, mtd_name in enumerate(methods):
            res_file_name = os.path.join(res_dir_model, os.path.basename(tar_files[tar_idx]).rsplit('.', 1)[0] + "_{}_{}.ply".format(mtd_name, template_density))

            # GPU
            V_tar_gpu = torch.tensor(V_tar, dtype=dtype, device=device).unsqueeze(0)
            N_tar_gpu = torch.tensor(N_tar, dtype=dtype, device=device).unsqueeze(0)
            V_src_gpu = torch.tensor(V_src, dtype=dtype, device=device)
            F_src_gpu = torch.tensor(F_src, dtype=torch.long, device=device)

            L = laplacian_uniform(V_src_gpu, F_src_gpu)

            mat = compute_matrix(V_src_gpu, F_src_gpu, 10)
            mat_mass = compute_matrix(V_src_gpu, F_src_gpu, None, alpha=0.5)
            u = to_differential(mat, V_src_gpu).clone()
            u.requires_grad_(True)

            optim = AdamUniform([u], lr=0.05)

            V_new = V_src_gpu.clone()

            for i in range(n_iter):
                tmp_V = from_differential(mat, u)
                src_mesh = Meshes(verts=tmp_V.unsqueeze(0), faces=F_src_gpu.unsqueeze(0))
                tmp_P, tmp_Np = sample_points_from_meshes(src_mesh, n_tar_verts * 4, True)
                
                x_nn = knn_points(V_tar_gpu, tmp_P, K=1)
                tmp_P_src = knn_gather(tmp_P, x_nn.idx).squeeze()
                tmp_N_src = knn_gather(tmp_Np, x_nn.idx).squeeze()
                p2p_loss = (V_tar_gpu.squeeze() - tmp_P_src).square().sum(-1).mean()
                p2pl_loss = ((V_tar_gpu.squeeze() - tmp_P_src) * tmp_N_src).sum(-1).square().mean()
                # normal_loss += (1 - torch.abs(torch.nn.functional.cosine_similarity(N_tar_gpu.squeeze(), tmp_N_src, dim=-1, eps=1e-6))).mean()
                normal_loss = (1 - torch.nn.functional.cosine_similarity(N_tar_gpu.squeeze(), tmp_N_src, dim=-1, eps=1e-12)).mean()
                
                alpha_cham = 0.0
                cham_loss = (alpha_cham * p2pl_loss + (1-alpha_cham) * p2p_loss)
                
                loss = cham_loss + normal_loss
                
                mass_mean_loss = torch.zeros(1)
                mass_lap_loss = torch.zeros(1)
                
                if mtd_name == 'ours' and i < n_iter * 0.5:
                    mass_mean_loss, mass_lap_loss = mass_loss(tmp_V, F_src_gpu, L, mat_mass)
                    
                    if i < n_iter * 0.25:
                        loss += mass_weight * (mass_mean_loss)
                    elif i < n_iter * 0.5:
                        # loss += mass_weight * (mass_mean_loss)
                        loss += 3 * mass_weight * (mass_lap_loss)
                
                losses[k,  i] = cham_loss.item()
                losses[k+2,i] = normal_loss.item()
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                print(""\
                            +"[{}/{}] ".format(str(tar_idx).zfill(4), len(tar_models))\
                            +"{}/{}, {}".format(tar_files[tar_idx].split("/")[-2], tar_files[tar_idx].split("/")[-1], mtd_name)\
                            +", iter [{}/{}]".format(i,n_iter)\
                            + ", cham_loss {:04.6f}".format(cham_loss.item())\
                            + ", normal_loss {:04.6f}".format(normal_loss.item())\
                            + ", mass_mean_loss {:04.6f}".format(mass_mean_loss.item())\
                            + ", mass_lap_loss {:04.6f}".format(mass_lap_loss.item())\
                        )

            # rescale
            with torch.no_grad():
                V_new = tmp_V.cpu().numpy()
            V_new *= scale_ori
            V_new += trans_ori[np.newaxis, :]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(V_new)
            mesh.triangles = o3d.utility.Vector3iVector(F_src)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(res_file_name, mesh)

        pd.DataFrame(data=losses.T, columns=table_label).to_csv(err_res_path)
