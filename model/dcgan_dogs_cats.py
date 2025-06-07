import torch
import torch.nn as nn
import lightning as L
from torchmetrics.image.fid import FrechetInceptionDistance
from gan import GAN
from data import PATH_DATASETS, BATCH_SIZE, NUM_WORKERS, CatsDogsDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvGenerator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, channels=3):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.to(self.device)

    def forward(self, z):
        return self.model(z.view(z.size(0), z.size(1), 1, 1).to(self.device))


class ConvDiscriminator(nn.Module):
    def __init__(self, ndf=64, channels=3):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
        )
        self.to(self.device)

    def forward(self, img):
        img = img.to(self.device)
        return self.model(img).view(img.size(0), -1)


class DCGAN(GAN):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 128,
        lr_g: float = 2e-4,
        lr_d: float = 4e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        ngf=64,
        ndf=64,
    ):
        super().__init__(channels=channels, width=width, height=height)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = ConvGenerator(latent_dim, ngf, channels)
        self.discriminator = ConvDiscriminator(ndf, channels)
        self.validation_z = torch.randn(8, self.hparams.latent_dim, device=device)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
        self.fid = FrechetInceptionDistance(normalize=True).to(device)

    def configure_optimizers(self):
        lr_g = self.hparams.lr_g
        lr_d = self.hparams.lr_d
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(b1, b2)
        )
        total_epochs = self.trainer.max_epochs
        start_decay = int(total_epochs * 0.8)
        sched_g = torch.optim.lr_scheduler.LambdaLR(
            opt_g,
            lr_lambda=lambda epoch: 1.0
            if epoch < start_decay
            else 1 - (epoch - start_decay) / (total_epochs - start_decay),
        )
        sched_d = torch.optim.lr_scheduler.LambdaLR(
            opt_d,
            lr_lambda=lambda epoch: 1.0
            if epoch < start_decay
            else 1 - (epoch - start_decay) / (total_epochs - start_decay),
        )
        return [opt_g, opt_d], [
            {"scheduler": sched_g, "interval": "epoch"},
            {"scheduler": sched_d, "interval": "epoch"},
        ]


if __name__ == "__main__":
    torch.cuda.empty_cache()
    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    data = CatsDogsDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name="DCGAN-dogs-cats",
        tracking_uri="./mlruns",
    )
    model = DCGAN(*data.dims, latent_dim=128, ngf=128, ndf=128)
    trainer = L.Trainer(
        max_epochs=500,
        logger=logger,
        precision="16-mixed",
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="fid_score",
                mode="min",
                dirpath="./checkpoints/dcgan-dogscats/",
                filename="dcgan-transforms-dogscats-{epoch:02d}-{fid_score:.2f}",
            ),
        ],
    )
    trainer.fit(model, datamodule=data)
