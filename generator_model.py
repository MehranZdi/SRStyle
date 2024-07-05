import torch
import torch.nn as nn

import config
from adain_model import AdaIN


adain_model = AdaIN()
adain_model.load_state_dict(torch.load('path/to/pretrained_adain.pth'))
adain_model.eval()
adain_model.to(config.DEVICE)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, use_in=True):
        super(ConvBlock, self).__init__()
        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.use_in = use_in
        self.in_norm = nn.InstanceNorm2d(out_channels) if use_in else None
        self.use_act = use_act
        self.act = nn.ReLU(inplace=True) if use_act else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.in_norm(x)
        if self.use_act:
            x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        return out + residual


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(ngf * 4) for _ in range(n_blocks)])

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, input, style_img):
        x = self.initial(input)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        content_features = self.final(x)
        styled_output = adain_model.generate(content_features, style_img)
        return styled_output

def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)


if __name__ == "__main__":
    test()