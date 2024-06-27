import torch
import torch.nn as nn
from adain_model import AdaIN  # Assuming AdaIN is defined in the imported module

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, use_in=False, **kwargs):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        ]
        if use_in:
            layers.append(nn.InstanceNorm2d(out_channels))
        if use_act:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, adain):
        super().__init__()
        self.adain1 = adain
        self.adain2 = adain
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1, use_in=False),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1, use_in=False),
        )

    def forward(self, x, style):
        out = self.block[0](x)
        out = self.adain(out, style)
        out = self.block[1](out)
        out = self.adain(out, style)
        return x + out

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9, adain):
        super().__init__()
        self.adain1 = adain
        self.adain2 = adain
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1, use_in=True),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1, use_in=True),
            ]
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_features * 4, self.adain) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1, use_in=False),
                ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1, use_in=False),
            ]
        )
        self.last = nn.Conv2d(
            num_features,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x, style):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        for res_block in self.res_blocks:
            x = res_block(x, style)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

def load_adain_weights(adain, path):
    adain.load_state_dict(torch.load(path))

def test():
    img_channels = 3
    img_size = 256
    style_dim = 512
    x = torch.randn((2, img_channels, img_size, img_size))
    style = torch.randn((2, style_dim))
    adain = AdaIN(style_dim, 512)
    load_adain_weights(adain, '/home/mehran/Git/SRStyle/model_state.pth')
    gen = Generator(img_channels, 9, adain=adain)
    print(gen(x, style).shape)

if __name__ == "__main__":
    test()