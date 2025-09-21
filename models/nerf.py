import torch
from torch import nn
import torch.nn.functional as F

from utils.util import get_voxel_vertices


class Embedding(nn.Module):
    """
    Embeds input to (x, sin(2^k x), cos(2^k x), ...).
    """

    def __init__(self,
                 in_channels,
                 n_freqs,
                 logscale=True):
        super().__init__()
        self.in_channels = in_channels
        self.n_freqs = n_freqs
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * n_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)

    def forward(self, x):
        """
        Inputs:
            x: Tensor of shape (B, in_channels)
        Outputs:
            Tensor of shape (B, out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, dim=-1)


class HashEmbedder(nn.Module):
    """Multi-resolution hash grid embedding."""

    def __init__(
        self,
        bounding_box,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
    ):
        super().__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = n_levels * n_features_per_level

        scale = (
            torch.log(self.finest_resolution)
            - torch.log(self.base_resolution)
        ) / (n_levels - 1)
        self.b = torch.exp(scale)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    2 ** self.log2_hashmap_size,
                    n_features_per_level,
                )
                for _ in range(n_levels)
            ]
        )
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, a=-0.0001, b=0.0001)

    def trilinear_interp(
        self,
        x,
        voxel_min_vertex,
        voxel_max_vertex,
        voxel_embedds,
    ):
        """Perform trilinear interpolation on the voxel embeddings."""
        # x: Tensor (B, 3)
        # voxel_min_vertex, voxel_max_vertex: Tensor (B, 3)
        # voxel_embedds: Tensor (B, 8, features)
        weights = (x - voxel_min_vertex) / (
            voxel_max_vertex - voxel_min_vertex
        )

        c00 = (
            voxel_embedds[:, 0] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 4] * weights[:, 0][:, None]
        )
        c01 = (
            voxel_embedds[:, 1] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 5] * weights[:, 0][:, None]
        )
        c10 = (
            voxel_embedds[:, 2] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 6] * weights[:, 0][:, None]
        )
        c11 = (
            voxel_embedds[:, 3] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 7] * weights[:, 0][:, None]
        )

        c0 = (
            c00 * (1 - weights[:, 1][:, None])
            + c10 * weights[:, 1][:, None]
        )
        c1 = (
            c01 * (1 - weights[:, 1][:, None])
            + c11 * weights[:, 1][:, None]
        )

        return (
            c0 * (1 - weights[:, 2][:, None])
            + c1 * weights[:, 2][:, None]
        )

    def forward(self, x):
        """Embed points with hash grid encoding."""
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            (
                vmin,
                vmax,
                hashed_indices,
                keep_mask,
            ) = get_voxel_vertices(
                x,
                self.bounding_box,
                resolution,
                self.log2_hashmap_size,
            )
            voxel_embeds = self.embeddings[i](hashed_indices)
            x_embed = self.trilinear_interp(
                x, vmin, vmax, voxel_embeds
            )
            x_embedded_all.append(x_embed)

        final_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
        embeds = torch.cat(x_embedded_all, dim=-1)
        return embeds, final_mask


class NeRF(nn.Module):
    """
    NeRF model that predicts density (sigma) and RGB.
    """

    def __init__(self,
                 depth=8,
                 width=256,
                 in_ch_xyz=63,
                 in_ch_dir=27,
                 skips=(4,)):
        super().__init__()
        self.depth = depth
        self.width = width
        self.in_ch_xyz = in_ch_xyz
        self.in_ch_dir = in_ch_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(depth):
            if i == 0:
                lin = nn.Linear(in_ch_xyz, width)
            elif i in skips:
                lin = nn.Linear(width + in_ch_xyz, width)
            else:
                lin = nn.Linear(width, width)
            block = nn.Sequential(lin, nn.ReLU(inplace=True))
            setattr(self, f"xyz_encoding_{i}", block)

        self.xyz_final = nn.Linear(width, width)

        # outputs
        self.sigma = nn.Linear(width, 1)
        self.rgb = nn.Sequential(
            nn.Linear(width + in_ch_dir, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x, sigma_only=False):
        """
        Encodes x and returns sigma (and rgb if requested).

        Inputs:
            x: Tensor of shape (B, in_ch_xyz [+ in_ch_dir])
            sigma_only: bool, if True returns only sigma
        Outputs:
            sigma or concat(rgb, sigma)
        """
        if sigma_only:
            xyz = x
        else:
            xyz, dirs = torch.split(
                x,
                [self.in_ch_xyz, self.in_ch_dir],
                dim=-1
            )

        h = xyz
        for i in range(self.depth):
            if i in self.skips:
                h = torch.cat((xyz, h), dim=-1)
            h = getattr(self, f"xyz_encoding_{i}")(h)

        sigma = self.sigma(h)
        if sigma_only:
            return sigma

        h_final = self.xyz_final(h)
        d_in = torch.cat((h_final, dirs), dim=-1)
        rgb = self.rgb(d_in)

        return torch.cat((rgb, sigma), dim=-1)
