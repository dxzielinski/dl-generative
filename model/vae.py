# vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance

from data import CatsDataModule, PATH_DATASETS, BATCH_SIZE, NUM_WORKERS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 8, 8)
        return self.deconv(x)


class VAE(L.LightningModule):
    def __init__(self, latent_dim=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(latent_dim=self.hparams.latent_dim)
        self.decoder = Decoder(latent_dim=self.hparams.latent_dim)
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.example_z = torch.randn(16, self.hparams.latent_dim, device=device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def compute_loss(self, x, x_hat, mu, logvar):
        recon = F.mse_loss(x_hat, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kld, recon, kld

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.to(device)
        x_hat, mu, logvar = self(x)
        loss, recon, kld = self.compute_loss(x, x_hat, mu, logvar)
        self.log_dict({"train_loss": loss, "train_recon": recon, "train_kld": kld})
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.to(device)
        x_hat, mu, logvar = self(x)
        loss, recon, kld = self.compute_loss(x, x_hat, mu, logvar)
        self.log_dict({"val_loss": loss, "val_recon": recon, "val_kld": kld})

        real = (x + 1) / 2
        fake = (x_hat + 1) / 2
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)

    def on_validation_epoch_end(self):
        fid_score = self.fid.compute()
        self.log("fid_score", fid_score)
        self.fid.reset()

        with torch.no_grad():
            samples = self.decoder(self.example_z)
            samples = (samples + 1) / 2
            grid = make_grid(samples, nrow=4)
            grid = grid.permute(1, 2, 0).cpu().numpy()
            plt.imshow(grid)
            plt.axis("off")
            plt.show()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    data = CatsDataModule(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS)
    data.setup("fit")

    model = VAE()

    trainer = L.Trainer(
        max_epochs=50,
        logger=L.pytorch.loggers.MLFlowLogger(
            experiment_name="VAE",
            tracking_uri="./mlruns",
        ),
        precision="16-mixed",
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="fid_score",
                mode="min",
                dirpath="./checkpoints/vae/",
                filename="vae-{epoch:02d}-{fid_score:.2f}",
            ),
        ],
    )

    trainer.fit(model, datamodule=data)
