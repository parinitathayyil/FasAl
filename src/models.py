import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2 if down else 1, padding=1, bias=False, padding_mode="reflect")
            if down else
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down1 = UNetBlock(features, features * 2, down=True, act="leaky")
        self.down2 = UNetBlock(features * 2, features * 4, down=True, act="leaky")
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 4, features * 4, 4, 2, 1), nn.ReLU(inplace=True)
        )

        self.up1 = UNetBlock(features * 4, features * 4, down=False, act="relu", use_dropout=True)
        self.up2 = UNetBlock(features * 8, features * 2, down=False, act="relu")
        self.up3 = UNetBlock(features * 4, features, down=False, act="relu")
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Restrict output pixels to range [0, 1]
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        bn = self.bottleneck(d3)
        
        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        return self.final_up(torch.cat([u3, d1], dim=1))

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256]):
        super().__init__()
        # Input takes concatenated [Low-Res Satellite Condition, High-Res Image]
        layers = []
        in_ch = in_channels * 2
        for feature in features:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, feature, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_ch = feature
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        return self.model(out)