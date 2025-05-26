import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

LATENT_DIM = 128
IMAGE_SIZE = 64
BATCH_SIZE = 128
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = r"C:\Users\Ksawery\Desktop\gitbub-repos\dl-generative\photos\cats\Data"

# ====== Model ======
class Encoder(nn.Module):
    def __init__(self, img_channels=3, latent_dim=LATENT_DIM):
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
    def __init__(self, img_channels=3, latent_dim=LATENT_DIM):
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

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, x_hat, x, mu, logvar):
        recon = F.mse_loss(x_hat, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kld, recon, kld

# ====== Visualization ======
def show_images(imgs, nrow=8, save_path=None):
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=nrow)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(nrow, 2))
    plt.axis("off")
    plt.imshow(grid)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ====== Dataset ======
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = ImageFolder(root=DATA_PATH, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ====== Training ======
def main():
    vae = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    losses = []

    for epoch in range(NUM_EPOCHS):
        vae.train()
        running_loss = 0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            x = x.to(DEVICE)
            x_hat, mu, logvar = vae(x)
            loss, recon, kld = vae.loss_function(x_hat, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")

        with torch.no_grad():
            z = torch.randn(16, LATENT_DIM).to(DEVICE)
            samples = vae.decoder(z)
            show_images(samples.cpu(), nrow=4)

    # Plot loss
    plt.plot(losses)
    plt.title("VAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
