import torch
import torch.nn as nn
from adain_model import AdaIN


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
        self.adain = adain
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, padding=1, use_in=False)
        self.conv2 = ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1, use_in=False)

    def forward(self, x, content_features, style_features):
        out = self.conv1(x)
        out = adain(out, style_features)  # Apply AdaIN using style features
        out = self.conv2(out)
        out = adain(out, style_features)  # Apply AdaIN using style features
        return x + out


class Generator(nn.Module):
    def __init__(self, adain, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.adain = AdaIN()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1, use_in=True),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1, use_in=True),
                # ConvBlock(num_features * 2, num_features * 2, kernel_size=3, stride=2, padding=1, use_in=True),
            ]
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_features * 4, adain) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1, use_in=True),
                ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1, use_in=True),
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

    def forward(self, x, content_img, style_img):
        # Extract features with VGGEncoder (512 channels)
        content_features = self.adain.vgg_encoder(content_img, output_last_feature=True)
        style_features = self.adain.vgg_encoder(style_img, output_last_feature=True)
        print(f'Content features shape: {content_features.shape}')
        print(f'Style features shape: {style_features.shape}')

        print(f'Initial input shape: {x.shape}')
        x = self.initial(x)
        print(f'After initial layer: {x.shape}')

        for i, layer in enumerate(self.down_blocks):
            x = layer(x)
            print(f'After down block {i}: {x.shape}')

        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, content_features, style_features)
            print(f'After residual block {i}: {x.shape}')

        for i, layer in enumerate(self.up_blocks):
            x = layer(x)
            print(f'After up block {i}: {x.shape}')

        x = torch.tanh(self.last(x))
        print(f'Final output shape: {x.shape}')
        return x


def load_adain_weights(adain, path):
    adain.load_state_dict(torch.load(path))


def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    content_images = torch.randn((2, img_channels, img_size, img_size))
    print(content_images.shape)
    style_images = torch.randn((2, img_channels, img_size, img_size))
    adain = AdaIN()
    load_adain_weights(adain, '/content/model_state.pth')
    gen = Generator(adain, img_channels, 128)
    print(gen(x, content_images, style_images).shape)


if __name__ == "__main__":
    test()