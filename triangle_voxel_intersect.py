import torch
import trimesh
import numpy as np
from tqdm.auto import tqdm


def triangle_aabb_voxel_intersect(
    vertices: torch.Tensor, faces: torch.Tensor, voxel_grid: torch.Tensor
):
    tri_min = vertices[faces].min(dim=1).values
    tri_max = vertices[faces].max(dim=1).values

    voxel_min = voxel_grid.min(dim=1).values
    voxel_max = voxel_grid.max(dim=1).values

    m_tri_min = tri_min.unsqueeze(1)  # (M, 1, 3)
    m_tri_max = tri_max.unsqueeze(1)  # (M, 1, 3)
    n_vox_min = voxel_min.unsqueeze(0)  # (1, N, 3)
    n_vox_max = voxel_max.unsqueeze(0)  # (1, N, 3)

    overlap_x = (m_tri_min[..., 0] <= n_vox_max[..., 0]) & (
        m_tri_max[..., 0] >= n_vox_min[..., 0]
    )
    overlap_y = (m_tri_min[..., 1] <= n_vox_max[..., 1]) & (
        m_tri_max[..., 1] >= n_vox_min[..., 1]
    )
    overlap_z = (m_tri_min[..., 2] <= n_vox_max[..., 2]) & (
        m_tri_max[..., 2] >= n_vox_min[..., 2]
    )

    aabb_intersect_mask = overlap_x & overlap_y & overlap_z
    intersecting_indices = torch.nonzero(aabb_intersect_mask)

    return intersecting_indices


def triangle_plane_voxel_intersect(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    voxel_grid: torch.Tensor,
    candidate_indices: torch.Tensor,
):
    """
    第二步：判断体素 AABB 是否与三角形所在的平面相交

    Args:
        vertices: [V, 3] 原始顶点
        faces: [M, 3] 索引
        voxel_grid: [N, 8, 3] 体素角点
        candidate_indices: [K, 2] 第一步 AABB 筛选出的 [tri_idx, vox_idx] 索引对

    Returns:
        refined_indices: [K', 2] 满足平面相交条件的索引对
    """
    if candidate_indices.shape[0] == 0:
        return candidate_indices


    tri_idx = candidate_indices[:, 0]
    vox_idx = candidate_indices[:, 1]


    v0 = vertices[faces[tri_idx, 0]]
    v1 = vertices[faces[tri_idx, 1]]
    v2 = vertices[faces[tri_idx, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = torch.cross(edge1, edge2, dim=-1)
    norm_len = torch.norm(normal, dim=-1, keepdim=True)
    normal = normal / (norm_len + 1e-8)

    v_min = voxel_grid[vox_idx].min(dim=1).values
    v_max = voxel_grid[vox_idx].max(dim=1).values

    center = (v_min + v_max) / 2.0
    h = (v_max - v_min) / 2.0
    dist_unnorm = torch.abs(torch.sum(normal * (center - v0), dim=-1))

    projection_radius = torch.sum(torch.abs(normal) * h, dim=-1)

    mask = dist_unnorm <= (projection_radius + 1e-6)

    return candidate_indices[mask]


def triangle_sat_edges_voxel_intersect(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    voxel_grid: torch.Tensor,
    candidate_indices: torch.Tensor,
):
    """
    第三步：执行 SAT 算法中剩下的 9 条边轴测试

    Args:
        vertices: [V, 3]
        faces: [M, 3]
        voxel_grid: [N, 8, 3]
        candidate_indices: [K, 2] 经过前两步筛选后的索引对
    """
    if candidate_indices.shape[0] == 0:
        return candidate_indices

    t_idx = candidate_indices[:, 0]
    v_idx = candidate_indices[:, 1]

    v0 = vertices[faces[t_idx, 0]]
    v1 = vertices[faces[t_idx, 1]]
    v2 = vertices[faces[t_idx, 2]]

    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    tri_edges = [e0, e1, e2]

    v_min = voxel_grid[v_idx].min(dim=1).values
    v_max = voxel_grid[v_idx].max(dim=1).values
    v_center = (v_min + v_max) / 2.0
    v_h = (v_max - v_min) / 2.0

    keep_mask = torch.ones(
        candidate_indices.shape[0], dtype=torch.bool, device=vertices.device
    )

    for e in tri_edges:
        axes = [
            torch.stack([torch.zeros_like(e[:, 0]), e[:, 2], -e[:, 1]], dim=-1),
            torch.stack([-e[:, 2], torch.zeros_like(e[:, 0]), e[:, 0]], dim=-1),
            torch.stack([e[:, 1], -e[:, 0], torch.zeros_like(e[:, 0])], dim=-1),
        ]

        for L in axes:
            L_norm = torch.norm(L, dim=-1, keepdim=True)
            valid_axis_mask = L_norm.squeeze(-1) > 1e-8 
            
            L = L / (L_norm + 1e-8)

            r = torch.sum(torch.abs(L) * v_h, dim=-1)
            p0 = torch.sum(L * (v0 - v_center), dim=-1)
            p1 = torch.sum(L * (v1 - v_center), dim=-1)
            p2 = torch.sum(L * (v2 - v_center), dim=-1)

            t_min = torch.min(torch.min(p0, p1), p2)
            t_max = torch.max(torch.max(p0, p1), p2)

            is_separated = valid_axis_mask & ((t_min > r + 1e-6) | (t_max < -r - 1e-6))
            
            keep_mask &= ~is_separated

            if not keep_mask.any():
                return candidate_indices[:0]

    return candidate_indices[keep_mask]


def get_full_voxel_grid(resolution: int):
    unit_voxel = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
    )
    grid = torch.arange(resolution)
    grid = torch.meshgrid(grid, grid, grid)
    grid = torch.stack(grid, dim=-1).reshape(-1, 1, 3)
    return grid + unit_voxel


def triangle_voxel_intersect(
    mesh: trimesh.Trimesh,
    voxel_grid: torch.Tensor,  # [N, 8, 3]
    chunk_size: int = 8192,
):
    device = voxel_grid.device
    vertices = torch.from_numpy(mesh.vertices).to(device).float()
    faces = torch.from_numpy(mesh.faces).to(device).long()

    N = voxel_grid.shape[0]
    all_hit_voxel_indices = []

    for start_v in tqdm(range(0, N, chunk_size), desc="Processing voxels", disable=True):
        end_v = min(start_v + chunk_size, N)

        curr_voxel_chunk = voxel_grid[start_v:end_v]

        intersecting_indices = triangle_aabb_voxel_intersect(
            vertices, faces, curr_voxel_chunk
        )

        if intersecting_indices.shape[0] == 0:
            continue

        refined_indices = triangle_plane_voxel_intersect(
            vertices, faces, curr_voxel_chunk, intersecting_indices
        )

        if refined_indices.shape[0] == 0:
            continue

        final_pairs_chunk = triangle_sat_edges_voxel_intersect(
            vertices, faces, curr_voxel_chunk, refined_indices
        )

        if final_pairs_chunk.shape[0] == 0:
            continue

        relative_hit_v_idx = final_pairs_chunk[:, 1].unique()
        global_hit_v_idx = relative_hit_v_idx + start_v

        all_hit_voxel_indices.append(global_hit_v_idx)

    if len(all_hit_voxel_indices) == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    final_indices = torch.cat(all_hit_voxel_indices)

    return final_indices


if __name__ == "__main__":
    from ocnn.octree import Octree, Points
    mesh = trimesh.load_remote("https://github.com/mikedh/trimesh/raw/main/models/bunny.ply").to_mesh()
    mesh.vertices = (mesh.vertices - mesh.vertices.min()) / (mesh.vertices.max() - mesh.vertices.min())
    mesh.export('mesh.obj')
    depth = 9
    full_depth = 1
    octree = Octree(depth=depth, full_depth=full_depth)
    unit_voxel = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]).reshape(1, 8, 3)
    for d in range(full_depth + 1):
        octree.octree_grow_full(d)
    for d in tqdm(range(full_depth, depth + 1), desc="Building octree"):
        mesh_scaled = mesh.copy()
        mesh_scaled.vertices = mesh_scaled.vertices * (2 ** d)
        x, y, z, b = octree.xyzb(d)
        voxel = torch.stack([x, y, z], dim=-1)
        voxel = voxel.reshape(-1, 1, 3) + unit_voxel
        idx = triangle_voxel_intersect(mesh_scaled, voxel)
        split = torch.zeros_like(octree.keys[d])
        split[idx] = 1
        octree.octree_split(split, d)
        octree.octree_grow(d + 1)
    octree_gt = Octree(depth=depth, full_depth=full_depth)
    points, _ = trimesh.sample.sample_surface(mesh, 10000000)
    octree_gt.build_octree(Points(torch.from_numpy(points * 2 - 1).float()))
    for d in range(depth + 1):
        gt_keys: torch.Tensor = octree_gt.keys[d]
        new_keys = octree.keys[d]
        set_gt = set(gt_keys.numpy().tolist())
        set_new = set(new_keys.numpy().tolist())
        print('depth: {}, num_gt: {}, num_new: {}, set_gt == set_new: {}, intersection: {}, difference: {}'.format(d, len(set_gt), len(set_new), set_gt == set_new, len(set_gt.intersection(set_new)), len(set_gt.difference(set_new))))

