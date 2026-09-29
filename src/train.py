import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import CropFusionDataset
from src.models import Generator, Discriminator
from src.metrics import MetricSuite

def train_pipeline(sat_path, drone_path, epochs=10, batch_size=4, lr=2e-4, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing pipeline on device: {device}")

    # Dataset & Loader
    dataset = CropFusionDataset(sat_path, drone_path, patch_size=256, num_patches=100)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models & Optimizers
    gen = Generator(in_channels=3).to(device)
    disc = Discriminator(in_channels=3).to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss Functions
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    metrics = MetricSuite(device=device)

    for epoch in range(1, epochs + 1):
        gen.train()
        disc.train()
        epoch_g_loss, epoch_d_loss = 0.0, 0.0

        for sat_img, drone_img in loader:
            sat_img, drone_img = sat_img.to(device), drone_img.to(device)

            # Train Discriminator
            fake_drone = gen(sat_img)
            disc_real = disc(sat_img, drone_img)
            disc_fake = disc(sat_img, fake_drone.detach())

            loss_d_real = bce_loss(disc_real, torch.ones_like(disc_real))
            loss_d_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake))
            loss_d = (loss_d_real + loss_d_fake) / 2

            opt_disc.zero_grad()
            loss_d.backward()
            opt_disc.step()

            # Train Generator
            disc_fake_updated = disc(sat_img, fake_drone)
            loss_g_gan = bce_loss(disc_fake_updated, torch.ones_like(disc_fake_updated))
            loss_g_l1 = l1_loss(fake_drone, drone_img) * 100.0  # L1 spatial recon weight
            loss_g = loss_g_gan + loss_g_l1

            opt_gen.zero_grad()
            loss_g.backward()
            opt_gen.step()

            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()

        # Evaluate Epoch Results
        eval_scores = metrics.compute_all(fake_drone, drone_img)
        print(f"Epoch [{epoch}/{epochs}] | Loss G: {epoch_g_loss/len(loader):.4f} | Loss D: {epoch_d_loss/len(loader):.4f} | PSNR: {eval_scores['PSNR_dB']} dB | SSIM: {eval_scores['SSIM']}")

        # Save Checkpoint
        if epoch % 5 == 0 or epoch == epochs:
            torch.save(gen.state_dict(), os.path.join(save_dir, f"generator_epoch_{epoch}.pth"))
            print(f"--> Saved checkpoint: {save_dir}/generator_epoch_{epoch}.pth")

if __name__ == "__main__":
    # Example local test execution call
    train_pipeline("data/raw_satellite/sample_sat.tif", "data/raw_drone/sample_drone.tif", epochs=2)