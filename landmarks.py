import torch
import trimesh
import numpy as np
import scipy.sparse as sparse

def raw_to_matrix(finds, bc, tF, num_input_vertices):
    """
    Convert a pair of face indices and barycentric coordinates into sparse matrix representation

    finds     (N1,) torch int
    bc        (N1 x 3) torch float
    tF         (#F x 3) torch int           face vertex indices for the mesh
    num_input_vertices           int

    returns)
    M       (N1 x num_input_vertices) torch csr sparse matrix float
    that satisfies
    (3d positions on the surface specified by finds & bc) = M @ tV
    """
    N1 = finds.shape[0]
    ii = torch.stack([torch.arange(N1, device=tF.device, dtype=torch.long) for _ in range(3)]).reshape(-1)
    jj = tF[finds].T.reshape(-1)
    data = bc.float().T.reshape(-1)

    i = torch.stack([ii, jj])
    v = data

    M = torch.sparse_coo_tensor(i, v, (N1, num_input_vertices))

    return M

def _get_position(finds, bc, tV, tF):
    if tV.ndim == 2:
        M = raw_to_matrix(finds, bc, tF, tV.shape[0])
        return M @ tV
    elif tV.ndim == 3:
        M = raw_to_matrix(finds, bc, tF, tV.shape[1])
        B = tV.shape[0]
        return torch.stack([M @ tV[b] for b in range(B)])
    else:
        raise RuntimeError(f'Unsupported tV dims {tV.shape}')

def _closest_surface_points(target, V, tF):
    with torch.no_grad():
        mesh = trimesh.Trimesh(V.cpu().numpy(), tF.cpu().numpy(), process=False)
        pts, _, finds = trimesh.proximity.closest_point(mesh, target.cpu().numpy())
        bc = trimesh.triangles.points_to_barycentric(mesh.triangles[finds], pts)
        finds = torch.from_numpy(finds).long().to(V.device)
        bc = torch.from_numpy(bc).to(V.device)
        return finds, bc
    
class SurfaceLandmarks:
    def __init__(self, tV, tF, pos_lmk, finds=None, bc=None, convert_landmarks_to_indices=False):
        self.tV = tV
        self.tF = tF
        if finds is None:
            self.reset_lmk(pos_lmk)
        else:
            self.finds = finds
            self.bc = bc
        
        if convert_landmarks_to_indices:
            bc_new = torch.zeros_like(self.bc)
            indices = torch.arange(self.bc.shape[0], device=self.bc.device, dtype=torch.long)
            bc_new[indices,self.bc.argmax(axis=-1)] = 1.0
            self.bc = bc_new
    
    def reset_lmk(self, V):
        with torch.no_grad():
            self.finds, self.bc = _closest_surface_points(V, self.tV, self.tF)
    
    def get_position(self, tV):
        return _get_position(self.finds, self.bc, tV, self.tF)

