import os
import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model, TaskType
from data import PATH_DATASETS, BATCH_SIZE, NUM_WORKERS, CatsDataModule


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FineTuneDiffusion(L.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        timesteps: int = 1000,
        lr: float = 1e-4,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()
        device = get_device()
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=torch.float32,
        )

        lora_config = LoraConfig(
            task_type=TaskType.UNET_2D,
            inference_mode=False,
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.scheduler = DDPMScheduler(num_train_timesteps=self.hparams.timesteps)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        imgs = imgs.to(self.device)
        noise = torch.randn_like(imgs)
        t = torch.randint(
            0,
            self.hparams.timesteps,
            (imgs.size(0),),
            device=self.device,
            dtype=torch.long,
        )
        noisy = self.scheduler.add_noise(imgs, noise, t)
        pred = self.unet(noisy, timesteps=t).sample
        loss = F.mse_loss(pred, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        imgs = imgs.to(self.device)
        x = torch.randn_like(imgs)
        for i in reversed(range(self.hparams.timesteps)):
            t = torch.full((imgs.size(0),), i, device=self.device, dtype=torch.long)
            eps = self.unet(x, timesteps=t).sample
            beta = self.scheduler.beta_schedule(torch.tensor([i])).to(self.device)
            alpha = 1 - beta
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha)) * eps)
            if i > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        gen = torch.clamp(x, -1.0, 1.0)
        real = (imgs + 1) / 2
        fake = (gen + 1) / 2
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)

    def on_validation_epoch_end(self):
        fid_score = self.fid.compute()
        self.log("fid_score", fid_score, prog_bar=True)
        self.fid.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.unet.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    L.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    data = CatsDataModule(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS)
    logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name="FineTuneDiffusion",
        tracking_uri="./mlruns",
    )
    model = FineTuneDiffusion(*data.dims)

    trainer = L.Trainer(
        max_epochs=10,
        logger=logger,
        precision="16-mixed",
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="fid_score",
                mode="min",
                dirpath="./checkpoints/diffusion_finetuned/",
                filename="ft-lora-{epoch:02d}-{fid_score:.2f}",
            )
        ],
    )
    trainer.fit(model, datamodule=data)
