import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


BOX_OFFSETS = torch.tensor(
    [
        [
            [i, j, k]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
    ],
    device="cuda",
)


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    Convert a depth map to a color image tensor.

    Args:
        depth (Tensor): depth map of shape (H, W).
        cmap (int): OpenCV colormap.

    Returns:
        Tensor: color image tensor of shape (3, H, W).
    """
    # To numpy and replace NaNs.
    depth_np = depth.cpu().numpy()
    depth_np = np.nan_to_num(depth_np)

    # Normalize to [0, 1].
    min_val = np.min(depth_np)
    max_val = np.max(depth_np)
    norm = (depth_np - min_val) / (max_val - min_val + 1e-8)

    # To 8‐bit and apply colormap.
    depth_8u = (norm * 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(depth_8u, cmap)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # PIL then to tensor.
    pil_img = Image.fromarray(color_rgb)
    tensor_img = T.ToTensor()(pil_img)

    return tensor_img


def hash_coords(coords, log2_hashmap_size):
    """Compute hashed indices for up to 7D coordinates.

    Args:
        coords (Tensor): Input coordinates, last dim ≤ 7.
        log2_hashmap_size (int): Log2 of hashmap size.

    Returns:
        Tensor: Hashed indices.
    """
    primes = [
        1,
        2654435761,
        805459861,
        3674653429,
        2097192037,
        1434869437,
        2165219737,
    ]
    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]
    mask = torch.tensor(
        (1 << log2_hashmap_size) - 1, device=xor_result.device)
    return xor_result & mask


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    """Get voxel vertices and hashed indices for 3D samples.

    Args:
        xyz (Tensor): Sample positions of shape (B, 3).
        bounding_box (tuple): (min_coords, max_coords).
        resolution (int or Tensor): Number of voxels per axis.
        log2_hashmap_size (int): Log2 of hashmap size.

    Returns:
        tuple: (voxel_min_vertex, voxel_max_vertex,
                hashed_voxel_indices, keep_mask).
    """
    box_min, box_max = bounding_box
    box_min = box_min.to(xyz.device)
    box_max = box_max.to(xyz.device)
    clipped_xyz = torch.clamp(xyz, min=box_min, max=box_max)
    keep_mask = xyz == clipped_xyz
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        xyz = clipped_xyz

    grid_size = (box_max - box_min) / resolution
    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).long()
    # voxel true location
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash_coords(voxel_indices, log2_hashmap_size)

    return (voxel_min_vertex,
            voxel_max_vertex,
            hashed_voxel_indices,
            keep_mask)
