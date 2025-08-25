import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T


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
