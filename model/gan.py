"""
Main model utilities for the project.
Esentially, it includes main models for image generation
and auxiliary functions.
"""

import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance

from data import CatsDataModule, PATH_DATASETS, BATCH_SIZE, NUM_WORKERS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.device = device

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )
        self.to(self.device)

    def forward(self, z):
        z = z.to(self.device)
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
        self.to(self.device)

    def forward(self, img):
        img = img.to(self.device)
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class GAN(L.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 128,
        lr: float = 2e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(
            latent_dim=self.hparams.latent_dim, img_shape=data_shape
        )
        self.discriminator = Discriminator(img_shape=data_shape)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
        self.fid = FrechetInceptionDistance(normalize=True).to(device)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch):
        imgs, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # log sampled images
        # sample_imgs = self.generated_imgs[:6]
        # grid = torchvision.utils.make_grid(sample_imgs).permute(1, 2, 0).cpu().numpy()
        # grid_scaled = (grid - grid.min()) / (grid.max() - grid.min())
        # step = self.global_step
        # run_id = self.logger.run_id
        # self.logger.experiment.log_image(
        #     image=grid_scaled, step=step, run_id=run_id, key="training_step_img"
        # )

        # ground truth result (ie: all fake)
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)

        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(
            self.discriminator(self.generated_imgs.detach()), fake
        )

        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        # log learning rates
        self.log("lr_g", optimizer_g.param_groups[0]["lr"], prog_bar=True)
        self.log("lr_d", optimizer_d.param_groups[0]["lr"], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        gen_imgs = self(z)
        real_imgs = (imgs + 1) / 2
        gen_imgs = (gen_imgs + 1) / 2
        self.fid.update(real_imgs, real=True)
        self.fid.update(gen_imgs, real=False)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        fid_score = self.fid.compute()
        self.log("fid_score", fid_score, prog_bar=True)
        self.fid.reset()
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z)
        sample_imgs = (sample_imgs + 1) / 2
        sample_imgs = torch.clamp(sample_imgs, 0, 1)
        grid = torchvision.utils.make_grid(sample_imgs)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        epoch = self.current_epoch
        run_id = self.logger.run_id
        self.logger.experiment.log_image(
            image=grid, step=epoch, run_id=run_id, key="validation_epoch_img"
        )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    data = CatsDataModule(
        data_dir=PATH_DATASETS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name="GAN",
        tracking_uri="./mlruns",
    )
    model = GAN(*data.dims)
    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    trainer = L.Trainer(
        max_epochs=150,
        logger=logger,
        precision="16-mixed",
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="fid_score",
                mode="min",
                dirpath="./checkpoints/gan/",
                filename="gan-transforms-{epoch:02d}-{d_loss:.2f}-{g_loss:.2f}-{fid_score:.2f}",
            ),
        ],
    )
    trainer.fit(model, datamodule=data)
