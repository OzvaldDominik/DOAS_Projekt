import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 100
SAVE_EVERY = 5
IMAGE_SIZE = (218, 178)
LAMBDA_RECON = 100
INPUT_DIR = 'mask_img'
TARGET_DIR = 'img_align_celeba'
OUTPUT_DIR = 'outputs'

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_filenames = sorted(os.listdir(INPUT_DIR))
train_files, temp_files = train_test_split(all_filenames, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

class ImagePairDataset(Dataset):
    def __init__(self, input_dir, target_dir, filenames, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        target_path = os.path.join(self.target_dir, self.filenames[idx])

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

transform = transforms.Compose([
    transforms.Resize((218, 178)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = ImagePairDataset(INPUT_DIR, TARGET_DIR, train_files, transform)
val_dataset = ImagePairDataset(INPUT_DIR, TARGET_DIR, val_files, transform)
test_dataset = ImagePairDataset(INPUT_DIR, TARGET_DIR, test_files, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # manji batch
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),   # 109 x 89
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # 55 x 45
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),# 28 x 23
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 56 x 46
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 112 x 92
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 224 x 184
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #print(x.type(), x.shape)
        x = F.interpolate(x, size=(218, 178), mode='bilinear', align_corners=False)
        #print("generator forward", x.shape)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(304128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

optimizer_G = optim.Adam(generator.parameters(), lr=2e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4)

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

for epoch in range(1, NUM_EPOCHS + 1):
    for i, (masked_imgs, real_imgs) in enumerate(train_loader):
        masked_imgs = masked_imgs.to(DEVICE)
        real_imgs = real_imgs.to(DEVICE)

        valid = torch.ones((masked_imgs.size(0), 1), device=DEVICE)
        fake = torch.zeros((masked_imgs.size(0), 1), device=DEVICE)

        # === Train Generator ===
        optimizer_G.zero_grad()
        gen_imgs = generator(masked_imgs)
        #print(gen_imgs.shape, real_imgs.shape)
        pred_fake = discriminator(gen_imgs)

        loss_G_adv = criterion_GAN(pred_fake, valid)
        #print(gen_imgs.shape, real_imgs.shape)
        loss_G_recon = criterion_L1(gen_imgs, real_imgs)
        loss_G = loss_G_adv + LAMBDA_RECON * loss_G_recon

        loss_G.backward()
        optimizer_G.step()

        # === Train Discriminator ===
        optimizer_D.zero_grad()
        pred_real = discriminator(real_imgs)
        loss_D_real = criterion_GAN(pred_real, valid)

        pred_fake = discriminator(gen_imgs.detach())
        loss_D_fake = criterion_GAN(pred_fake, fake)

        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        if i % 50 == 0:
            print(f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}/{len(train_loader)}] "
                  f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")


    if epoch % SAVE_EVERY == 0:
        generator.eval()
        with torch.no_grad():
            for j, (masked_imgs, real_imgs) in enumerate(val_loader):
                masked_imgs = masked_imgs.to(DEVICE)
                real_imgs = real_imgs.to(DEVICE)

                gen_imgs = generator(masked_imgs)

                # Spremi prvu validacijsku batch sliku
                imgs_concat = torch.cat([masked_imgs, gen_imgs, real_imgs], dim=0)
                save_image(imgs_concat * 0.5 + 0.5, f"{OUTPUT_DIR}/val_epoch_{epoch}.png", nrow=4)
                break  # samo prvi batch
        generator.train()

print("Evaluating on test set...")
generator.eval()
with torch.no_grad():
    for i, (masked_imgs, real_imgs) in enumerate(test_loader):
        masked_imgs = masked_imgs.to(DEVICE)
        real_imgs = real_imgs.to(DEVICE)
        gen_imgs = generator(masked_imgs)

        imgs_concat = torch.cat([masked_imgs, gen_imgs, real_imgs], dim=0)
        save_image(imgs_concat * 0.5 + 0.5, f"{OUTPUT_DIR}/test_batch_{i}.png", nrow=4)
        if i == 2: break  # primi 3 batcha
