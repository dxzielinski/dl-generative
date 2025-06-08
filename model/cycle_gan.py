"""
It is not ready yet
"""

import itertools

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn.functional as F
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance


class CycleGANDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataroot: str = "datasets/horse2zebra/",
        batch_size: int = 1,
        image_size: int = 256,
        input_nc: int = 3,
        output_nc: int = 3,
        num_workers: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        transforms_ = [
            transforms.Resize(int(self.hparams.image_size * 1.12), Image.BICUBIC),
            transforms.RandomCrop(self.hparams.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.dataset = ImageDataset(
            self.hparams.dataroot, transforms_=transforms_, unaligned=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )


class CycleGAN(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.0002,
        n_epochs: int = 200,
        decay_epoch: int = 100,
        input_nc: int = 3,
        output_nc: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Networks
        self.netG_A2B = Generator(input_nc, output_nc)
        self.netG_B2A = Generator(output_nc, input_nc)
        self.netD_A = Discriminator(input_nc)
        self.netD_B = Discriminator(output_nc)

        # Initialization
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        # Buffers
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # FID metric
        self.fid = FrechetInceptionDistance(normalize=True).to(self.device)
        # fixed noise for sampling
        self.validation_z = torch.randn(
            8, self.hparams.input_nc * 0
        )  # placeholder, length unused for CycleGAN

    def forward(self, x, direction="A2B"):
        if direction == "A2B":
            return self.netG_A2B(x)
        return self.netG_B2A(x)

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(
            itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            lr=self.hparams.lr,
            betas=(0.5, 0.999),
        )
        opt_D_A = torch.optim.Adam(
            self.netD_A.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999)
        )
        opt_D_B = torch.optim.Adam(
            self.netD_B.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999)
        )

        lr_lambda = LambdaLR(
            self.hparams.n_epochs,
            start_epoch=0,
            decay_epoch=self.hparams.decay_epoch,
        ).step
        sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_lambda)
        sched_D_A = torch.optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda=lr_lambda)
        sched_D_B = torch.optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda=lr_lambda)

        return (
            [opt_G, opt_D_A, opt_D_B],
            [
                {"scheduler": sched_G, "interval": "epoch"},
                {"scheduler": sched_D_A, "interval": "epoch"},
                {"scheduler": sched_D_B, "interval": "epoch"},
            ],
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A = batch["A"]
        real_B = batch["B"]
        target_real = torch.ones(real_A.size(0), device=self.device)
        target_fake = torch.zeros(real_A.size(0), device=self.device)

        # Generator
        if optimizer_idx == 0:
            same_B = self.netG_A2B(real_B)
            loss_id_B = self.criterion_identity(same_B, real_B) * 5.0
            same_A = self.netG_B2A(real_A)
            loss_id_A = self.criterion_identity(same_A, real_A) * 5.0

            fake_B = self.netG_A2B(real_A)
            loss_GAN_A2B = self.criterion_GAN(self.netD_B(fake_B), target_real)

            fake_A = self.netG_B2A(real_B)
            loss_GAN_B2A = self.criterion_GAN(self.netD_A(fake_A), target_real)

            recov_A = self.netG_B2A(fake_B)
            loss_cycle_ABA = self.criterion_cycle(recov_A, real_A) * 10.0
            recov_B = self.netG_A2B(fake_A)
            loss_cycle_BAB = self.criterion_cycle(recov_B, real_B) * 10.0

            loss_G = (
                loss_id_A
                + loss_id_B
                + loss_GAN_A2B
                + loss_GAN_B2A
                + loss_cycle_ABA
                + loss_cycle_BAB
            )
            self.log("loss_G", loss_G, prog_bar=True)
            return loss_G

        # Discriminator A
        if optimizer_idx == 1:
            loss_D_real = self.criterion_GAN(self.netD_A(real_A), target_real)
            fake_A = self.fake_A_buffer.push_and_pop(self.netG_B2A(real_B))
            loss_D_fake = self.criterion_GAN(self.netD_A(fake_A.detach()), target_fake)
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            self.log("loss_D_A", loss_D_A, prog_bar=True)
            return loss_D_A

        # Discriminator B
        if optimizer_idx == 2:
            loss_D_real = self.criterion_GAN(self.netD_B(real_B), target_real)
            fake_B = self.fake_B_buffer.push_and_pop(self.netG_A2B(real_A))
            loss_D_fake = self.criterion_GAN(self.netD_B(fake_B.detach()), target_fake)
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            self.log("loss_D_B", loss_D_B, prog_bar=True)
            return loss_D_B

    def validation_step(self, batch, batch_idx):
        real_A = batch["A"]
        real_B = batch["B"]
        # generate for both directions
        fake_B = self.netG_A2B(real_A)
        fake_A = self.netG_B2A(real_B)
        # rescale
        real_A = (real_A + 1) / 2
        real_B = (real_B + 1) / 2
        fake_A = (fake_A + 1) / 2
        fake_B = (fake_B + 1) / 2
        # update FID on both
        self.fid.update(real_A, real=True)
        self.fid.update(fake_A, real=False)
        self.fid.update(real_B, real=True)
        self.fid.update(fake_B, real=False)

    def on_validation_epoch_end(self):
        fid_score = self.fid.compute()
        self.log("fid_score", fid_score, prog_bar=True)
        self.fid.reset()

        # sample and log images
        num_samples = 8
        z_A = torch.randn(num_samples, self.hparams.input_nc, device=self.device)
        z_B = torch.randn(num_samples, self.hparams.output_nc, device=self.device)
        samples_A2B = self.netG_A2B(z_A)
        samples_B2A = self.netG_B2A(z_B)
        samples = torch.cat([samples_A2B, samples_B2A], dim=0)
        samples = (samples + 1) / 2
        grid = torchvision.utils.make_grid(samples, nrow=num_samples)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        epoch = self.current_epoch
        run_id = self.logger.run_id
        self.logger.experiment.log_image(
            image=grid, step=epoch, run_id=run_id, key="validation_epoch_img"
        )


if __name__ == "__main__":
    dm = CycleGANDataModule()
    model = CycleGAN()
    trainer = Trainer(max_epochs=model.hparams.n_epochs)
    trainer.fit(model, datamodule=dm)
