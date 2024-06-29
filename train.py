import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from discriminator_model import Discriminator
from generator_model import Generator
from dataset import ImageDataset
from tqdm import tqdm
from adain_model import AdaIN



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def train(disc_L, disc_H, gen_H, gen_L, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    writer = SummaryWriter()
    lr_reals = 0
    lr_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (hr_image, lr_image) in enumerate(loop):
        hr_image = hr_image.to(config.DEVICE)
        lr_image = lr_image.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake_lr = gen_L(hr_image)
            D_L_real = disc_L(lr_image)
            D_L_fake = disc_L(fake_lr.detack())
            lr_reals += D_L_real.mean().item()
            lr_fakes += D_L_fake.mean().item()
            D_L_real_loss = mse(D_L_real, torch.ones_like(D_L_real))
            D_L_fake_loss = mse(D_L_fake, torch.zeros_like(D_L_fake))
            D_L_loss = D_L_real_loss + D_L_fake_loss

            fake_hr = gen_H(lr_image)
            D_H_real = disc_H(hr_image)
            D_H_fake = disc_H(fake_hr.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            D_loss = (D_L_loss + D_H_loss)

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()


        with torch.cuda.amp.autocast():
            D_L_fake = disc_L(fake_lr)
            D_H_fake = disc_H(fake_hr)
            loss_G_L = mse(D_L_fake, torch.ones_like(D_L_fake))
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))

            # cycle loss
            cycle_H = gen_H(fake_lr)
            cycle_L = gen_L(fake_hr)
            cycle_H_loss = l1(hr_image, cycle_H)
            cycle_L_loss = l1(lr_image, cycle_L)

            # add all together
            G_loss = (
                    loss_G_H
                    + loss_G_L
                    + cycle_H_loss * config.LAMBDA_CYCLE
                    + cycle_L_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Log losses to TensorBoard
        writer.add_scalar('Loss/Discriminator', D_loss.item(), epoch * len(loader) + idx)
        writer.add_scalar('Loss/Generator', G_loss.item(), epoch * len(loader) + idx)
        writer.add_scalar('Loss/D_L_real', D_L_real_loss.item(), epoch * len(loader) + idx)
        writer.add_scalar('Loss/D_L_fake', D_L_fake_loss.item(), epoch * len(loader) + idx)
        writer.add_scalar('Loss/D_H_real', D_H_real_loss.item(), epoch * len(loader) + idx)
        writer.add_scalar('Loss/D_H_fake', D_H_fake_loss.item(), epoch * len(loader) + idx)
        writer.add_scalar('Loss/Cycle_H', cycle_H_loss.item(), epoch * len(loader) + idx)
        writer.add_scalar('Loss/Cycle_L', cycle_L_loss.item(), epoch * len(loader) + idx)

        if idx % 200 == 0:
            save_image(fake_lr * 0.5 + 0.5, f"saved_images/lr_{idx}.png")
            save_image(fake_hr * 0.5 + 0.5, f"saved_images/hr_{idx}.png")

        loop.set_postfix(lr_real=lr_reals / (idx + 1), lr_fake=lr_fakes / (idx + 1))

def main():
    adain = AdaIN()
    adain.load_state_dict(torch.load('/home/mehran/Git/SRStyle/model_state.pth'))

    disc_L = Discriminator(in_channels=3).to(config.DEVICE)     #for the real LR and fake LR
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)     #for the real HR and fake HR
    gen_H = Generator(adain, img_channels=3, num_residuals=9).to(config.DEVICE)     #
    gen_L = Generator(adain, img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_L.parameters()) + list(disc_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    opt_gen = optim.Adam(
        list(gen_L.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    train_hr_dir = '/home/mehran/SRStyle_Dataset/train/Styled'
    train_lr_dirs = ['home/mehran/SRStyle_Dataset/train/low_res_x4', 'home/mehran/SRStyle_Dataset/train/low_res_x3']

    val_hr_dir = 'home/mehran/SRStyle_Dataset/validation/Styled'
    val_lr_dirs = ['/home/mehran/SRStyle_Dataset/validation/x4', '/home/mehran/SRStyle_Dataset/validation/x3']

    dataset = ImageDataset(
        train_hr_dir,
        train_lr_dirs,
        transform=config.transforms,
    )
    val_dataset = ImageDataset(
        val_hr_dir,
        val_lr_dirs,
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir='runs/cyclegan_experiment')

    for epoch in range(config.EPOCHS):
        train(
            disc_L,
            disc_H,
            gen_H,
            gen_L,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

    if config.SAVE_MODEL:
        save_checkpoint(gen_L, opt_gen, filename=config.CHECKPOINT_GEN_L)
        save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
        save_checkpoint(disc_L, opt_disc, filename=config.CHECKPOINT_CRITIC_L)
        save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)

    writer.close()


if __name__ == "__main__":
    main()
