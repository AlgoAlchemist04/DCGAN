import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
import os
import torchvision.utils as vutils

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e6} MB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e6} MB")
else:
    print("Using CPU. WARNING: This will be significantly slower!")

# Hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 500
FEATURES_DISC = 64
FEATURES_GEN = 64
CHECKPOINT_PATH = "gan_checkpoint.pth"
GEN_IMAGES_DIR = "generated_images"
os.makedirs(GEN_IMAGES_DIR, exist_ok=True)

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * CHANNELS_IMG, [0.5] * CHANNELS_IMG),
])

# Dataset and DataLoader
dataset = datasets.ImageFolder(root="moon_dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize models and optimizers
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0
start_epoch = 0

# Function to save checkpoint
def save_checkpoint(epoch):
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": gen.state_dict(),
        "discriminator_state_dict": disc.state_dict(),
        "optimizer_G_state_dict": opt_gen.state_dict(),
        "optimizer_D_state_dict": opt_disc.state_dict()
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved at epoch {epoch}")

# Function to load checkpoint
def load_checkpoint():
    global start_epoch
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        gen.load_state_dict(checkpoint["generator_state_dict"])
        disc.load_state_dict(checkpoint["discriminator_state_dict"])
        opt_gen.load_state_dict(checkpoint["optimizer_G_state_dict"])
        opt_disc.load_state_dict(checkpoint["optimizer_D_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Loaded checkpoint from epoch {start_epoch - 1}")
    else:
        print("No checkpoint found, starting fresh.")

# Load checkpoint if exists
load_checkpoint()

gen.train()
disc.train()

# Training Loop
for epoch in range(start_epoch, NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        # Train Discriminator
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real, device=device))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake, device=device))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output, device=device))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # GPU Memory Check
        if device.type == "cuda" and batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f} | "
                  f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e6} MB | "
                  f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e6} MB")

            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            # Save generated images
            vutils.save_image(fake[:32], f"{GEN_IMAGES_DIR}/epoch_{epoch}.png", normalize=True)
            print(f"Saved generated images for epoch {epoch} in {GEN_IMAGES_DIR}")
            
            step += 1
    
    # Save checkpoint after every epoch
    save_checkpoint(epoch)
