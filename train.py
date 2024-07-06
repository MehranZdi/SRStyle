import itertools
import os
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch
from torchvision import transforms
from generator_model import Generator
import torch.nn.functional as F
from discriminator_model import Discriminator
import config
from adain_model import VGGEncoder, calc_mean_std

def compute_content_loss(generated, content):
    generated_features = VGGEncoder(generated)
    content_features = VGGEncoder(content)
    return F.mse_loss(generated_features, content_features)

def compute_style_loss(generated, style):
    generated_features = VGGEncoder(generated)
    style_features = VGGEncoder(style)
    style_loss = 0
    for gf, sf in zip(generated_features, style_features):
        gm, gs = calc_mean_std(gf)
        sm, ss = calc_mean_std(sf)
        style_loss += F.mse_loss(gm, sm) + F.mse_loss(gs, ss)
    return style_loss

def train_cycle_gan_with_adain(generator_G, generator_F, discriminator_D_X, discriminator_D_Y, dataloader, num_epochs=200, alpha=1.0, lambda_cycle=10, lambda_id=5, lambda_style=10):
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = torch.optim.Adam(itertools.chain(generator_G.parameters(), generator_F.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_X = torch.optim.Adam(discriminator_D_X.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_Y = torch.optim.Adam(discriminator_D_Y.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, batch in enumerate(tqdm(dataloader)):
            real_X, real_Y, style_img = batch

            # Move data to device
            real_X = real_X.to(device)
            real_Y = real_Y.to(device)
            style_img = style_img.to(device)

            # Update Generators
            optimizer_G.zero_grad()

            # Generate images with AdaIN
            fake_Y = generator_G(real_X, style_img)
            rec_X = generator_F(fake_Y, real_X)

            fake_X = generator_F(real_Y, style_img)
            rec_Y = generator_G(fake_X, real_Y)

            # Adversarial loss
            loss_GAN_G = criterion_GAN(discriminator_D_Y(fake_Y), torch.ones_like(discriminator_D_Y(fake_Y)))
            loss_GAN_F = criterion_GAN(discriminator_D_X(fake_X), torch.ones_like(discriminator_D_X(fake_X)))

            # Cycle consistency loss
            loss_cycle_X = criterion_cycle(rec_X, real_X)
            loss_cycle_Y = criterion_cycle(rec_Y, real_Y)

            # Identity loss
            loss_identity_G = criterion_identity(generator_G(real_Y, real_Y), real_Y)
            loss_identity_F = criterion_identity(generator_F(real_X, real_X), real_X)

            # Style transfer loss
            content_loss = compute_content_loss(fake_Y, real_X)
            style_loss = compute_style_loss(fake_Y, style_img)

            # Total generator loss
            loss_G = loss_GAN_G + loss_GAN_F + lambda_cycle * (loss_cycle_X + loss_cycle_Y) + lambda_id * (loss_identity_G + loss_identity_F) + lambda_style * (content_loss + style_loss)
            loss_G.backward()
            optimizer_G.step()

            # Update Discriminators
            optimizer_D_X.zero_grad()
            loss_D_X_real = criterion_GAN(discriminator_D_X(real_X), torch.ones_like(discriminator_D_X(real_X)))
            loss_D_X_fake = criterion_GAN(discriminator_D_X(fake_X.detach()), torch.zeros_like(discriminator_D_X(fake_X.detach())))
            loss_D_X = (loss_D_X_real + loss_D_X_fake) * 0.5
            loss_D_X.backward()
            optimizer_D_X.step()

            optimizer_D_Y.zero_grad()
            loss_D_Y_real = criterion_GAN(discriminator_D_Y(real_Y), torch.ones_like(discriminator_D_Y(real_Y)))
            loss_D_Y_fake = criterion_GAN(discriminator_D_Y(fake_Y.detach()), torch.zeros_like(discriminator_D_Y(fake_Y.detach())))
            loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5
            loss_D_Y.backward()
            optimizer_D_Y.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss G: {loss_G.item()}, Loss D_X: {loss_D_X.item()}, Loss D_Y: {loss_D_Y.item()}, Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}')


# Example usage
generator_G = Generator(input_nc=3, output_nc=3).to(config.DEVICE)
generator_F = Generator(input_nc=3, output_nc=3).to(config.DEVICE)
discriminator_D_X = Discriminator(input_nc=3).to(config.DEVICE)
discriminator_D_Y = Discriminator(input_nc=3).to(config.DEVICE)

# Prepare dataloader
class IntegratedDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, style_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.style_dir = style_dir
        self.lr_images = os.listdir(lr_dir)
        self.hr_images = os.listdir(hr_dir)
        self.style_images = os.listdir(style_dir)
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image = Image.open(os.path.join(self.lr_dir, self.lr_images[idx])).convert('RGB')
        hr_image = Image.open(os.path.join(self.hr_dir, self.hr_images[idx])).convert('RGB')
        style_image = Image.open(os.path.join(self.style_dir, self.style_images[idx % len(self.style_images)])).convert('RGB')

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
            style_image = self.transform(style_image)

        return lr_image, hr_image, style_image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = IntegratedDataset(lr_dir='path/to/low_resolution_images', hr_dir='path/to/high_resolution_images', style_dir='path/to/wikiart_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# Train the integrated model
train_cycle_gan_with_adain(generator_G, generator_F, discriminator_D_X, discriminator_D_Y, dataloader, num_epochs=200)
