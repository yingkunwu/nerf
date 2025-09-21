import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .ray_utils import get_ray_directions, get_rays


# This 3D box define the extremes of the camera viewing frustums in world
# coordinates.
def get_bbox3d(camera_transforms, W, H, near=2.0, far=6.0):
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    directions = get_ray_directions(H, W, focal)

    all_points = []

    for frame in camera_transforms["frames"]:
        c2w = torch.tensor(frame["transform_matrix"])[:3, :4]
        rays_o, rays_d = get_rays(directions, c2w)

        # only take image corners
        corner_ids = [0, W-1, H*W-W, H*W-1]
        rays_o_c = rays_o[corner_ids]
        rays_d_c = rays_d[corner_ids]

        # near and far plane points for these rays
        pts_near = rays_o_c + near * rays_d_c
        pts_far = rays_o_c + far * rays_d_c

        all_points.append(pts_near)
        all_points.append(pts_far)

    all_points = torch.cat(all_points, dim=0)  # (num_frames*8, 3)

    min_bound = all_points.min(dim=0).values
    max_bound = all_points.max(dim=0).values

    # pad a bit
    pad = torch.tensor([1.0, 1.0, 1.0])
    return min_bound - pad, max_bound + pad


class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], (
            'image width must equal image height!'
        )
        self.img_wh = img_wh
        self.white_back = True

        self.transform = T.ToTensor()
        self.read_meta()

    def read_meta(self):
        meta_path = os.path.join(
            self.root_dir, f'transforms_{self.split}.json'
        )
        with open(meta_path, 'r') as meta_file:
            self.meta = json.load(meta_file)

        w, h = self.img_wh
        angle_x = self.meta['camera_angle_x']
        self.focal = 0.5 * 800 / np.tan(0.5 * angle_x)
        self.focal *= self.img_wh[0] / 800

        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        self.directions = get_ray_directions(h, w, self.focal)

        if self.split == 'train':
            self._prepare_train_data()

        self.bbox = get_bbox3d(self.meta, *self.img_wh, self.near, self.far)

    def _prepare_train_data(self):
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []

        for frame in self.meta['frames']:
            pose = np.array(frame['transform_matrix'])[:3, :4]
            self.poses.append(pose)
            c2w = torch.FloatTensor(pose)

            image_path = os.path.join(
                self.root_dir, f'{frame["file_path"]}.png'
            )
            self.image_paths.append(image_path)

            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)
            img = img.view(4, -1).permute(1, 0)
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
            self.all_rgbs.append(img)

            rays_o, rays_d = get_rays(self.directions, c2w)
            near = self.near * torch.ones_like(rays_o[:, :1])
            far = self.far * torch.ones_like(rays_o[:, :1])
            rays = torch.cat([rays_o, rays_d, near, far], dim=1)
            self.all_rays.append(rays)

        self.all_rays = torch.cat(self.all_rays, dim=0)
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)

    def __len__(self):
        if self.split == 'train':
            # Iterate only 10% of the data per epoch
            return max(1, int(0.1 * len(self.all_rays)))
        if self.split == 'val':
            return 8
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train':
            return {
                'rays': self.all_rays[idx * 9],
                'rgbs': self.all_rgbs[idx * 9],
            }

        frame = self.meta['frames'][idx]
        c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
        image_path = os.path.join(
            self.root_dir, f'{frame["file_path"]}.png'
        )
        img = Image.open(image_path)
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)
        valid_mask = img[-1].flatten() > 0
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        rays_o, rays_d = get_rays(self.directions, c2w)
        near = self.near * torch.ones_like(rays_o[:, :1])
        far = self.far * torch.ones_like(rays_o[:, :1])
        rays = torch.cat([rays_o, rays_d, near, far], dim=1)

        return {
            'rays': rays,
            'rgbs': img,
            'c2w': c2w,
            'valid_mask': valid_mask,
        }


if __name__ == "__main__":
    dataset = BlenderDataset(
        root_dir='data/nerf_synthetic/lego', split='train', img_wh=(800, 800))
    print(f"Number of samples in {dataset.split} set: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample rays shape: {sample['rays'].shape}, "
          f"Sample rgbs shape: {sample['rgbs'].shape}")
