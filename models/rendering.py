import torch
from einops import rearrange


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample N_importance samples from bins with distribution defined by weights.
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)
    inds_g = torch.stack([below, above], dim=-1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_g).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_g).view(N_rays, N_importance, 2)

    # Linear interpolation inside the bin:
    # map u from its CDF interval to the corresponding z-interval.
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1
    samples = bins_g[..., 0] + (
        (u - cdf_g[..., 0]) / denom
    ) * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def inference(model,
              embeddings,
              xyz_,
              dir_,
              z_vals,
              chunk,
              noise_std,
              white_back,
              weights_only=False):
    """
    Perform NeRF model inference on sampled points.

    Args:
        model: NeRF model (coarse or fine).
        embeddings: positional embedding module for rgb and dirs.
        xyz_: tensor, shape (N_rays, N_samples, 3).
        dir_: tensor, shape (N_rays, 3).
        z_vals: tensor, shape (N_rays, N_samples).
        chunk: int, chunk size for inference.
        noise_std: float, std for noise added to sigma.
        white_back: bool, whether to use white background.
        weights_only: bool, if True return only weights.

    Returns:
        If weights_only:
            weights: (N_rays, N_samples)
        else:
            rgb_final: (N_rays, 3)
            depth_final: (N_rays,)
            weights: (N_rays, N_samples)
    """
    N_rays, N_samples_ = xyz_.shape[:2]
    embedding_xyz, embedding_dir = embeddings

    dir_embed = embedding_dir(dir_)

    # Flatten points
    xyz_flat = xyz_.reshape(-1, 3)
    if not weights_only:
        dir_embed_rep = torch.repeat_interleave(
            dir_embed, repeats=N_samples_, dim=0
        )

    # Inference in chunks
    out_chunks = []
    B = xyz_flat.shape[0]
    for i in range(0, B, chunk):
        xyz_emb = embedding_xyz(xyz_flat[i:i + chunk])
        if weights_only:
            inp = xyz_emb
        else:
            inp = torch.cat(
                [xyz_emb, dir_embed_rep[i:i + chunk]], dim=1
            )
        out_chunks.append(model(inp, sigma_only=weights_only))
    out = torch.cat(out_chunks, dim=0)

    if weights_only:
        sigmas = out.view(N_rays, N_samples_)
    else:
        rgbsigma = out.view(N_rays, N_samples_, 4)
        rgbs = rgbsigma[..., :3]
        sigmas = rgbsigma[..., 3]

    # Volume rendering
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_inf], dim=-1)
    deltas = deltas * torch.norm(dir_, dim=-1, keepdim=True)

    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))
    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10],
        dim=-1
    )
    weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[:, :-1]

    if weights_only:
        return weights
    weights_sum = weights.sum(dim=1)

    # sum over n_sample dimension
    rgb_final = torch.sum(weights[..., None] * rgbs, dim=1)
    depth_final = torch.sum(weights * z_vals, dim=1)

    # ignore background
    if white_back:
        rgb_final = rgb_final + (1 - weights_sum[..., None])

    return rgb_final, depth_final, weights


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024 * 32,
                white_back=False,
                test_time=False):
    """
    Render rays by computing NeRF model outputs on rays.

    Args:
        models: list of NeRF models [coarse, fine].
        embeddings: [positional embedding, directional embedding].
        rays: tensor, shape (N_rays, 8), (o, d, near, far).
        N_samples: int, number of coarse samples.
        use_disp: bool, sample in disparity space.
        perturb: float, perturb factor.
        noise_std: float, sigma noise std.
        N_importance: int, number of fine samples.
        chunk: int, chunk size for inference.
        white_back: bool, white background.
        test_time: bool, inference only for coarse.

    Returns:
        dict with keys:
            'rgb_coarse', 'depth_coarse', 'opacity_coarse',
            optionally 'rgb_fine', 'depth_fine', 'opacity_fine'.
    """
    model_coarse = models[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    near, far = rays[:, 6:7], rays[:, 7:8]
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)

    # z_vals.shape -> [num_rays, N_samples]
    if not use_disp:
        z_vals = near * (1 - z_steps) + far * z_steps
    else:
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)
    z_vals = z_vals.expand(rays.shape[0], N_samples)

    if perturb > 0:
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * rand

    pts_coarse = (
        rearrange(rays_o, 'r d -> r 1 d')
        + rearrange(rays_d, 'r d -> r 1 d')
        * rearrange(z_vals, 'r n -> r n 1')
    )

    if test_time:
        weights_coarse = inference(
            model_coarse, embeddings, pts_coarse, rays_d, z_vals,
            chunk, noise_std, white_back, weights_only=True
        )
        return {'opacity_coarse': weights_coarse.sum(dim=1)}

    rgb_coarse, depth_coarse, weights_coarse = inference(
        model_coarse, embeddings, pts_coarse, rays_d, z_vals,
        chunk, noise_std, white_back, weights_only=False
    )
    result = {
        'rgb_coarse': rgb_coarse,
        'depth_coarse': depth_coarse,
        'opacity_coarse': weights_coarse.sum(dim=1)
    }
    if N_importance > 0:
        model_fine = models[1]
        # To build a piecewise-constant PDF, we need bin centers.
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        z_fine = sample_pdf(
            mids, weights_coarse[:, 1:-1], N_importance,
            det=(perturb == 0)
        ).detach()
        z_vals = torch.sort(torch.cat([z_vals, z_fine], dim=-1), dim=-1)[0]

        pts_fine = (
            rearrange(rays_o, 'r d -> r 1 d')
            + rearrange(rays_d, 'r d -> r 1 d')
            * rearrange(z_vals, 'r n -> r n 1')
        )
        rgb_fine, depth_fine, weights_fine = inference(
            model_fine, embeddings, pts_fine, rays_d, z_vals,
            chunk, noise_std, white_back, weights_only=False
        )
        result.update({
            'rgb_fine': rgb_fine,
            'depth_fine': depth_fine,
            'opacity_fine': weights_fine.sum(dim=1)
        })
    return result
