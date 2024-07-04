import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import config
from generator_model import Generator
from discriminator_model import Discriminator

def train_cycle_gan_with_style(generator_G, generator_F, discriminator_D_X, discriminator_D_Y, dataloader, adain_model, num_epochs=200, alpha=1.0, lambda_cycle=10, lambda_id=5, lambda_style=10):
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
            real_X = real_X.to(config.DEVICE)
            real_Y = real_Y.to(config.DEVICE)
            style_img = style_img.to(config.DEVICE)

            # Update Generators
            optimizer_G.zero_grad()

            # Generate images
            fake_Y = generator_G(real_X)
            rec_X = generator_F(fake_Y)

            fake_X = generator_F(real_Y)
            rec_Y = generator_G(fake_X)

            # Adversarial loss
            loss_GAN_G = criterion_GAN(discriminator_D_Y(fake_Y), torch.ones_like(discriminator_D_Y(fake_Y)))
            loss_GAN_F = criterion_GAN(discriminator_D_X(fake_X), torch.ones_like(discriminator_D_X(fake_X)))

            # Cycle consistency loss
            loss_cycle_X = criterion_cycle(rec_X, real_X)
            loss_cycle_Y = criterion_cycle(rec_Y, real_Y)

            # Identity loss
            loss_identity_G = criterion_identity(generator_G(real_Y), real_Y)
            loss_identity_F = criterion_identity(generator_F(real_X), real_X)

            # Style transfer loss
            styled_fake_Y = adain_model.generate(fake_Y, style_img, alpha)
            content_loss = compute_content_loss(styled_fake_Y, fake_Y)
            style_loss = compute_style_loss(styled_fake_Y, style_img)

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