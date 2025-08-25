import torch

from .blender import BlenderDataset


def build_dataset(cfg, image_set):
    assert image_set in ["train", "val"], \
        f"Invalid image_set: {image_set}. Must be train or val."
    if cfg.dataset.name == 'blender':
        dataset = BlenderDataset(
            root_dir=cfg.dataset.root_dir,
            split=image_set,
            img_wh=cfg.dataset.img_wh
        )
    else:
        raise NotImplementedError(
            f"Dataset {cfg.dataset.name} not implemented.")

    batch_size = cfg.batch_size if image_set == "train" else 1
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(image_set == "train"),
            num_workers=cfg.num_workers,
            pin_memory=True
        )
