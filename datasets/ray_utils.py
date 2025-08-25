import torch
from typing import Union


def create_meshgrid(H: int,
                    W: int,
                    normalized_coordinates: bool = True,
                    dtype: torch.dtype = None,
                    device: Union[torch.device, str] = None) -> torch.Tensor:
    """
    Create a meshgrid in the Kornia format.

    Returns:
        grid: (1, H, W, 2) where grid[..., 0] = x, grid[..., 1] = y.
              If normalized_coordinates:
                  x in [-1, 1] across width, y in [-1, 1] across height.
              Else:
                  x in [0, W-1], y in [0, H-1].
    """
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, W, dtype=dtype, device=device)
        ys = torch.linspace(-1.0, 1.0, H, dtype=dtype, device=device)
    else:
        xs = torch.linspace(0, W - 1, W, dtype=dtype, device=device)
        ys = torch.linspace(0, H - 1, H, dtype=dtype, device=device)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2) -> [..., 0]=x, [..., 1]=y
    return grid.unsqueeze(0)  # (1, H, W, 2)


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference:
      https://www.scratchapixel.com/lessons/3d-basic-rendering/
      ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coord.
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)

    # Assumes a simple pinhole camera with square pixels, zero skew, and the
    # principal point at image center.
    # Under those assumptions the intrinsic matrix is:
    # K =
    # [[f & 0 & W / 2]
    #  [0 & f & H / 2]
    #  [0 & 0 & 1    ]]

    directions = torch.stack(
        [
            (i - W / 2) / focal,
            -(j - H / 2) / focal,
            -torch.ones_like(i),
        ],
        dim=-1,
    )

    return directions


def get_rays(directions, c2w):
    """
    Get ray origins and normalized directions in world coord.
    Reference:
      https://www.scratchapixel.com/lessons/3d-basic-rendering/
      ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3)
        c2w: (3, 4) camera-to-world transform

    Outputs:
        rays_o: (H*W, 3)
        rays_d: (H*W, 3)
    """
    # Rotate ray directions to world coord
    rays_d = directions @ c2w[:, :3].T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Origin is camera origin in world coord
    rays_o = c2w[:, 3].expand(rays_d.shape)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coord to NDC.
    Reference:
      http://www.songho.ca/opengl/gl_projectionmatrix.html
      https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    Inputs:
        H, W, focal: image h, w and focal
        near: (N_rays) or float, near plane depth
        rays_o: (N_rays, 3), ray origins in world coord
        rays_d: (N_rays, 3), ray dirs in world coord

    Outputs:
        rays_o: (N_rays, 3), in NDC
        rays_d: (N_rays, 3), in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * ox_oz
    o1 = -1.0 / (H / (2.0 * focal)) * oy_oz
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (W / (2.0 * focal)) * (
        rays_d[..., 0] / rays_d[..., 2] - ox_oz
    )
    d1 = -1.0 / (H / (2.0 * focal)) * (
        rays_d[..., 1] / rays_d[..., 2] - oy_oz
    )
    d2 = 1.0 - o2

    rays_o = torch.stack([o0, o1, o2], dim=-1)
    rays_d = torch.stack([d0, d1, d2], dim=-1)

    return rays_o, rays_d
