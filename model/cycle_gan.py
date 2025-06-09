"""
It is not ready yet
"""

import itertools
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance

from dcgan import ConvDiscriminator, ConvGenerator
from data import CycleGANDataModule


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, (
            "Empty buffer or trying to create a black hole. Be careful."
        )
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.autograd.Variable(torch.cat(to_return))


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class CycleGAN(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.0002,
        input_nc: int = 3,
        output_nc: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.netG_A2B = ConvGenerator(input_nc, output_nc)
        self.netG_B2A = ConvGenerator(output_nc, input_nc)
        self.netD_A = ConvDiscriminator(input_nc)
        self.netD_B = ConvDiscriminator(output_nc)

        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        self.fid = FrechetInceptionDistance(normalize=True).to(self.device)

        self.validation_z = torch.randn(8, self.hparams.input_nc * 0)

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
        total_epochs = self.trainer.max_epochs
        start_decay = int(total_epochs * 0.8)
        sched_G = torch.optim.lr_scheduler.LambdaLR(
            opt_G,
            lr_lambda=lambda epoch: 1.0
            if epoch < start_decay
            else 1 - (epoch - start_decay) / (total_epochs - start_decay),
        )
        sched_D_A = torch.optim.lr_scheduler.LambdaLR(
            opt_D_A,
            lr_lambda=lambda epoch: 1.0
            if epoch < start_decay
            else 1 - (epoch - start_decay) / (total_epochs - start_decay),
        )
        sched_D_B = torch.optim.lr_scheduler.LambdaLR(
            opt_D_B,
            lr_lambda=lambda epoch: 1.0
            if epoch < start_decay
            else 1 - (epoch - start_decay) / (total_epochs - start_decay),
        )

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

        if optimizer_idx == 1:
            loss_D_real = self.criterion_GAN(self.netD_A(real_A), target_real)
            fake_A = self.fake_A_buffer.push_and_pop(self.netG_B2A(real_B))
            loss_D_fake = self.criterion_GAN(self.netD_A(fake_A.detach()), target_fake)
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            self.log("loss_D_A", loss_D_A, prog_bar=True)
            return loss_D_A

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

        fake_B = self.netG_A2B(real_A)
        fake_A = self.netG_B2A(real_B)

        real_A = (real_A + 1) / 2
        real_B = (real_B + 1) / 2
        fake_A = (fake_A + 1) / 2
        fake_B = (fake_B + 1) / 2

        self.fid.update(real_A, real=True)
        self.fid.update(fake_A, real=False)
        self.fid.update(real_B, real=True)
        self.fid.update(fake_B, real=False)

    def on_validation_epoch_end(self):
        fid_score = self.fid.compute()
        self.log("fid_score", fid_score, prog_bar=True)
        self.fid.reset()

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

    def test_step(self, batch, batch_idx):
        real_A = batch["A"]
        real_B = batch["B"]
        fake_B = self.netG_A2B(real_A)
        fake_A = self.netG_B2A(real_B)
        real_A = (real_A + 1) / 2
        real_B = (real_B + 1) / 2
        fake_A = (fake_A + 1) / 2
        fake_B = (fake_B + 1) / 2
        self.fid.update(real_A, real=True)
        self.fid.update(fake_A, real=False)
        self.fid.update(real_B, real=True)
        self.fid.update(fake_B, real=False)


if __name__ == "__main__":
    data = CycleGANDataModule()
    model = CycleGAN()
    trainer = Trainer(max_epochs=model.hparams.n_epochs)
    trainer.fit(model, datamodule=data)
