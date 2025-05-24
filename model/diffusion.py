import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics.image.fid import FrechetInceptionDistance
from data import PATH_DATASETS, BATCH_SIZE, NUM_WORKERS, CatsDataModule
from torchvision import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Block(nn.Module):
    """
    A simple Conv block with time embedding
    """

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.silu(h)
        time_emb = self.mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.conv2(h)
        h = self.bn2(h)
        return F.silu(h)


class SimpleUNet(nn.Module):
    """
    A lightweight U-Net for diffusion with skip projections
    """

    def __init__(self, in_ch, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.down1 = Block(in_ch, base_ch, time_emb_dim)
        self.down2 = Block(base_ch, base_ch * 2, time_emb_dim)
        self.down3 = Block(base_ch * 2, base_ch * 4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.skip_proj2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=1)
        self.skip_proj1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=1)
        self.up1 = Block(base_ch * 4, base_ch * 2, time_emb_dim)
        self.up2 = Block(base_ch * 2, base_ch, time_emb_dim)
        self.conv_last = nn.Conv2d(base_ch, in_ch, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)
        u1 = F.interpolate(d3, scale_factor=2, mode="nearest")
        d2_proj = self.skip_proj2(d2)
        u1 = self.up1(u1 + d2_proj, t_emb)
        u2 = F.interpolate(u1, scale_factor=2, mode="nearest")
        d1_proj = self.skip_proj1(d1)
        u2 = self.up2(u2 + d1_proj, t_emb)
        return self.conv_last(u2)


class Diffusion(L.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        lr: float = 1e-4,
        batch_size: int = BATCH_SIZE,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleUNet(in_ch=channels)
        self.automatic_optimization = False
        betas = torch.linspace(
            self.hparams.beta_start,
            self.hparams.beta_end,
            self.hparams.timesteps,
            device=device,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod)
        )
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.validation_z = torch.randn(8, channels, height, width, device=device)

    def q_sample(self, x0, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x0)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om_acp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_acp * x0 + sqrt_om_acp * noise

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        imgs = imgs.to(device)
        batch_size = imgs.size(0)
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(imgs)
        x_noisy = self.q_sample(imgs, t, noise)
        pred_noise = self.model(x_noisy, t)
        loss = F.mse_loss(pred_noise, noise)
        self.log("train_loss", loss, prog_bar=True)
        opt = self.optimizers()
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        real = (imgs + 1) / 2
        x = torch.randn_like(imgs).to(device)
        for i in reversed(range(self.hparams.timesteps)):
            t = torch.full((imgs.size(0),), i, device=device, dtype=torch.long)
            eps = self.model(x, t)
            beta = self.betas[t].view(-1, 1, 1, 1)
            alpha = 1 - beta
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha)) * eps)
            if i > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        gen = torch.clamp(x, 0.0, 1.0)
        self.fid.update(real, real=True)
        self.fid.update(gen, real=False)
        return gen

    def on_validation_epoch_end(self):
        fid_score = self.fid.compute()
        self.log("fid_score", fid_score, prog_bar=True)
        self.fid.reset()
        with torch.no_grad():
            x = self.validation_z.clone()
            for i in reversed(range(self.hparams.timesteps)):
                t = torch.full((x.size(0),), i, device=device, dtype=torch.long)
                eps = self.model(x, t)
                beta = self.betas[t].view(-1, 1, 1, 1)
                alpha = 1 - beta
                x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha)) * eps)
                if i > 0:
                    x = x + torch.sqrt(beta) * torch.randn_like(x)
            samples = torch.clamp(x, 0.0, 1.0)
        grid = utils.make_grid(samples)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        self.logger.experiment.log_image(
            image=grid,
            step=self.current_epoch,
            run_id=self.logger.run_id,
            key="validation_epoch_img",
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    data = CatsDataModule(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS)
    logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name="Diffusion",
        tracking_uri="./mlruns",
    )
    model = Diffusion(*data.dims, timesteps=50)
    trainer = L.Trainer(
        max_epochs=50,
        logger=logger,
        precision="16-mixed",
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="fid_score",
                mode="min",
                dirpath="./checkpoints/diffusion/",
                filename="diffusion-{epoch:02d}-{fid_score:.2f}",
            ),
        ],
    )
    trainer.fit(model, datamodule=data)
