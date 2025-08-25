import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.utils import save_image
from collections import defaultdict

from models import NeRF, Embedding, render_rays
from datasets import build_dataset
from utils.metrics import psnr
from utils.loss import MSELoss
from utils.util import visualize_depth

# Faster, but less precise
torch.set_float32_matmul_precision("high")
# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)


class NERFModule(LightningModule):
    def __init__(self, cfg, log_dir):
        super().__init__()
        self.cfg = cfg
        self.embedding_xyz = Embedding(3, cfg.model.xyz_embed_dim)
        self.embedding_dir = Embedding(3, cfg.model.dir_embed_dim)
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF(
            depth=cfg.model.depth,
            width=cfg.model.width,
            in_ch_xyz=3 + 3 * cfg.model.xyz_embed_dim * 2,
            in_ch_dir=3 + 3 * cfg.model.dir_embed_dim * 2,
            skips=cfg.model.skips,
        )
        self.models = [self.nerf_coarse]
        if cfg.N_importance > 0:
            self.nerf_fine = NeRF(
                depth=cfg.model.depth,
                width=cfg.model.width,
                in_ch_xyz=3 + 3 * cfg.model.xyz_embed_dim * 2,
                in_ch_dir=3 + 3 * cfg.model.dir_embed_dim * 2,
                skips=cfg.model.skips,
            )
            self.models += [self.nerf_fine]

        self.loss = MSELoss()

        self.save_vis_path = os.path.join(log_dir, cfg.save_vis_path)
        os.makedirs(self.save_vis_path, exist_ok=True)
        self.save_hyperparameters(cfg)

    def decode_batch(self, batch):
        rays = batch['rays']  # (B, 8)
        rgbs = batch['rgbs']  # (B, 3)
        return rays, rgbs

    def configure_optimizers(self):
        parameters = []
        for model in self.models:
            parameters += list(model.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.cfg.scheduler.decay_step,
            gamma=self.cfg.scheduler.decay_gamma
        )

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.cfg.chunk):
            results_chunk = render_rays(
                self.models,
                self.embeddings,
                rays[i:i+self.cfg.chunk],
                self.cfg.N_samples,
                self.cfg.use_disp,
                self.cfg.perturb,
                self.cfg.noise_std,
                self.cfg.N_importance,
                self.cfg.chunk,
                self.cfg.dataset.white_back)

            for k, v in results_chunk.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results

    def training_step(self, batch, batch_idx):
        rays, rgbs = self.decode_batch(batch)
        results = self.forward(rays)

        log = {'train/loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            log['train/psnr'] = psnr(results[f'rgb_{typ}'], rgbs)

        self.log_dict(
            log,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.cfg.batch_size)

        return {'loss': log['train/loss']}

    def validation_step(self, batch, batch_idx):
        rays, rgbs = self.decode_batch(batch)
        # batch size is 1 for val
        rays = rays.squeeze(0)
        rgbs = rgbs.squeeze(0)
        results = self.forward(rays)

        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_idx == 0:
            W, H = self.cfg.dataset.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))
            stack = torch.stack([img_gt, img, depth])
            self.logger.experiment.add_images(
                'val/visualization', stack, self.global_step)

            save_name = os.path.join(
                self.save_vis_path, f'val_{self.global_step:06d}.png')
            save_image(stack, save_name, nrow=3)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        return log


@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def run(cfg: DictConfig):
    # instead of print(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    # replace WandbLogger with TensorBoard
    logger = TensorBoardLogger(
        save_dir="./logs",
        name=cfg.run_name
    )
    run_root = logger.log_dir

    train_loader = build_dataset(cfg, "train")
    val_loader = build_dataset(cfg, "val")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(run_root, "weight"),
        filename="best",
        monitor='train/psnr',
        mode='max',
        save_top_k=1,
        save_last=True
    )
    callbacks = [lr_monitor, ckpt_cb]

    trainer = Trainer(
        accelerator='gpu',
        devices=[cfg.gpu],
        precision=32,
        max_epochs=cfg.num_epochs,
        gradient_clip_val=0.1,
        deterministic=False,
        num_sanity_val_steps=1,
        logger=logger,
        callbacks=callbacks
    )

    module = NERFModule(cfg, log_dir=run_root)
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    run()
