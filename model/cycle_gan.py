"""
Got inspired by: https://github.com/aitorzip/PyTorch-CycleGAN
Modified to work well with Lightning and to use FID as a metric.
"""

import itertools
import random
import torch
import torch.nn as nn

import lightning as L

from torchmetrics.image.fid import FrechetInceptionDistance

from data import CycleGANDataModule


import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


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


class CycleGAN(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.0002,
        input_nc: int = 3,
        output_nc: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.netG_A2B = Generator(input_nc, output_nc)
        self.netG_B2A = Generator(output_nc, input_nc)
        self.netD_A = Discriminator(input_nc)
        self.netD_B = Discriminator(output_nc)

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

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            real_A = batch["A"]
            real_B = batch["B"]
        else:
            real_A, real_B = batch
        target_real = torch.ones(real_A.size(0), device=self.device).unsqueeze(1)
        target_fake = torch.zeros(real_A.size(0), device=self.device).unsqueeze(1)

        opt_G, opt_D_A, opt_D_B = self.optimizers()

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

        opt_G.zero_grad()
        self.manual_backward(loss_G)
        opt_G.step()
        self.log("loss_G", loss_G, prog_bar=True)

        self.log("lr_G", opt_G.param_groups[0]["lr"], prog_bar=True)

        loss_D_real = self.criterion_GAN(self.netD_A(real_A), target_real)
        fake_A_buff = self.fake_A_buffer.push_and_pop(self.netG_B2A(real_B))
        loss_D_fake = self.criterion_GAN(self.netD_A(fake_A_buff.detach()), target_fake)
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        opt_D_A.zero_grad()
        self.manual_backward(loss_D_A)
        opt_D_A.step()
        self.log("loss_D_A", loss_D_A, prog_bar=True)
        loss_D_real = self.criterion_GAN(self.netD_B(real_B), target_real)
        fake_B_buff = self.fake_B_buffer.push_and_pop(self.netG_A2B(real_A))
        loss_D_fake = self.criterion_GAN(self.netD_B(fake_B_buff.detach()), target_fake)
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5

        opt_D_B.zero_grad()
        self.manual_backward(loss_D_B)
        opt_D_B.step()
        self.log("loss_D_B", loss_D_B, prog_bar=True)

        return loss_G

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            real_A = batch["A"]
            real_B = batch["B"]
        else:
            real_A, real_B = batch

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


if __name__ == "__main__":
    torch.cuda.empty_cache()
    L.seed_everything(42)
    logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name="CycleGAN",
        tracking_uri="./mlruns",
    )
    data = CycleGANDataModule(
        data_dir="/home/dxzielinski/Downloads/dogs-vs-cats-cycle", batch_size=64
    )
    model = CycleGAN()
    torch.set_float32_matmul_precision("medium")
    trainer = L.Trainer(
        max_epochs=50,
        logger=logger,
        precision="16-mixed",
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="fid_score",
                mode="min",
                dirpath="./checkpoints/cycle-gan/",
                filename="cycle-gan-{epoch:02d}-{fid_score:.2f}",
            ),
        ],
    )
    trainer.fit(model, datamodule=data)
