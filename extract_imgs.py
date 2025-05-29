import torch
import os
import cv2
import numpy as np
from model import Generator  # Import your model class

# Load model
model_path = "gan_checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channels_noise = 100  # This is usually 100 in DCGAN
channels_img = 3      # 3 for RGB images, 1 for grayscale
features_g = 64       # Number of features in generator

model = Generator(channels_noise, channels_img, features_g).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Generate images
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for i in range(10):  # Generate 10 images
        noise = torch.randn(1, latent_dim, 1, 1, device=device)  # Adjust input if needed
        fake_image = model(noise).squeeze().cpu().numpy()
        fake_image = (fake_image * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, f"generated_image_{i}.png"), fake_image)

print(f"Saved images in {output_dir}")
