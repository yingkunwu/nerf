import os
import torch
from omegaconf import OmegaConf
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import imageio

from models import NeRF, Embedding, HashEmbedder, render_rays
from datasets.ray_utils import get_ray_directions, get_rays
from datasets import build_dataset

# Faster, but less precise
torch.set_float32_matmul_precision("high")


def trans_t(t: float) -> torch.Tensor:
    """Translate along the z-axis by t."""
    m = torch.eye(4, dtype=torch.float32)
    m[2, 3] = t
    return m


def rot_phi(phi: float) -> torch.Tensor:
    """Rotate around the x-axis by phi radians."""
    phi_t = torch.tensor(phi, dtype=torch.float32)
    m = torch.eye(4, dtype=torch.float32)
    cos_phi = torch.cos(phi_t)
    sin_phi = torch.sin(phi_t)
    m[1, 1] = cos_phi
    m[1, 2] = -sin_phi
    m[2, 1] = sin_phi
    m[2, 2] = cos_phi
    return m


def rot_theta(theta: float) -> torch.Tensor:
    """Rotate around the y-axis by theta radians."""
    theta_t = torch.tensor(theta, dtype=torch.float32)
    m = torch.eye(4, dtype=torch.float32)
    cos_t = torch.cos(theta_t)
    sin_t = torch.sin(theta_t)
    m[0, 0] = cos_t
    m[0, 2] = -sin_t
    m[2, 0] = sin_t
    m[2, 2] = cos_t
    return m


def pose_spherical(theta: float,
                   phi: float,
                   radius: float) -> torch.Tensor:
    """Get camera-to-world transformation matrix."""
    c2w = trans_t(radius)
    angle = phi / 180.0 * torch.pi
    c2w = rot_phi(angle) @ c2w
    angle = theta / 180.0 * torch.pi
    c2w = rot_theta(angle) @ c2w
    flip = torch.tensor(
        [
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return flip @ c2w


def load_ckpt(model, ckpt_path, mode_name):
    ckpt = torch.load(ckpt_path, weights_only=False)['state_dict']
    # Filter checkpoint entries to only those matching the mode_name prefix
    prefix = f"{mode_name}."
    model_state = {
        k[len(prefix):]: v
        for k, v in ckpt.items()
        if k.startswith(prefix)
    }
    model.load_state_dict(model_state, strict=True)


@torch.no_grad()
def batched_inference(models, embeddings, rays, N_samples, N_importance,
                      use_disp, chunk, white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)

    return results


if __name__ == "__main__":
    cfg = {
        "img_wh": [800, 800],
        "chunk": 80000,
        "N_samples": 64,
        "N_importance": 128,
        "near": 2.0,
        "far": 6.0,
        "use_disp": False,
        "white_back": True,
        "gpu": 0,
        "num_workers": 4,
        # model architecture
        "model": {
            "depth": 6,
            "width": 64,
            "dir_embed_dim": 4,
            "skips": [3],
            "n_levels": 16,  # for hash embedding
            "n_features_per_level": 2,  # for hash embedding
            "log2_hashmap_size": 19,  # for hash embedding
            "base_resolution": 16,  # for hash embedding
            "finest_resolution": 512,  # for hash embedding
        },
        "dataset": {
            "name": "blender",
            "root_dir": "data/nerf_synthetic/lego",
            "img_wh": [800, 800],
            "white_back": True,
        },
        "ckpt_path": "logs/exp/version_101/weight/last.ckpt",
    }
    # convert the plain dict to an OmegaConf object
    cfg = OmegaConf.create(cfg)

    val_loader = build_dataset(cfg, "val")
    bbox = val_loader.dataset.bbox

    embedding_xyz = HashEmbedder(
        bounding_box=bbox,
        n_levels=cfg.model.n_levels,
        n_features_per_level=cfg.model.n_features_per_level,
        log2_hashmap_size=cfg.model.log2_hashmap_size,
        base_resolution=cfg.model.base_resolution,
        finest_resolution=cfg.model.finest_resolution
    ).cuda().eval()
    embedding_dir = Embedding(3, cfg.model.dir_embed_dim)
    nerf_coarse = NeRF(
        depth=cfg.model.depth,
        width=cfg.model.width,
        in_ch_xyz=cfg.model.n_levels * cfg.model.n_features_per_level,
        in_ch_dir=3 + 3 * cfg.model.dir_embed_dim * 2,
        skips=cfg.model.skips,
    )
    load_ckpt(nerf_coarse, cfg.ckpt_path, "nerf_coarse")
    load_ckpt(embedding_xyz, cfg.ckpt_path, "embedding_xyz")
    nerf_coarse.cuda().eval()
    models = [nerf_coarse]

    if cfg.N_importance > 0:
        nerf_fine = NeRF(
            depth=cfg.model.depth,
            width=cfg.model.width,
            in_ch_xyz=cfg.model.n_levels * cfg.model.n_features_per_level,
            in_ch_dir=3 + 3 * cfg.model.dir_embed_dim * 2,
            skips=cfg.model.skips,
        )
        load_ckpt(nerf_fine, cfg.ckpt_path, "nerf_fine")
        nerf_fine.cuda().eval()
        models += [nerf_fine]

    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    # derive the output dir from the checkpoint path
    ckpt_dir = os.path.dirname(cfg.ckpt_path)
    exp_dir = os.path.dirname(ckpt_dir)
    dir_name = os.path.join(exp_dir, "video")
    os.makedirs(dir_name, exist_ok=True)

    w, h = cfg.img_wh
    ray_dirs = get_ray_directions(800, 800, 800)  # (H, W, 3)
    near = cfg.near * torch.ones((h * w, 1))
    far = cfg.far * torch.ones((h * w, 1))
    for i, th in enumerate(tqdm(np.linspace(0., 360., 120, endpoint=False))):
        c2w = pose_spherical(th, -30., 4.)
        rays_o, rays_d = get_rays(ray_dirs, c2w[:3, :4])
        rays = torch.cat([rays_o, rays_d, near, far], dim=1).cuda()
        results = batched_inference(
            models,
            embeddings,
            rays,
            cfg.N_samples,
            cfg.N_importance,
            cfg.use_disp,
            cfg.chunk,
            cfg.white_back)

        # get numpy arrays
        img_pred = results['rgb_coarse'].view(h, w, 3).cpu().numpy()
        depth_pred = results['depth_coarse'].view(h, w).cpu().numpy()

        # normalize depth to [0,1]
        dmin, dmax = depth_pred.min(), depth_pred.max()
        depth_norm = (depth_pred - dmin) / (dmax - dmin + 1e-8)

        # turn into 3‐channel uint8
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_img = np.stack([depth_uint8]*3, axis=-1)

        # convert rgb to uint8
        img_uint8 = (img_pred * 255).astype(np.uint8)

        # concatenate horizontally
        img = np.concatenate([img_uint8, depth_img], axis=1)

        imgs += [img]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img)

    imageio.mimsave(os.path.join(dir_name, 'output.gif'), imgs, fps=30)
