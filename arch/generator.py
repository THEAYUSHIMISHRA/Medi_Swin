import torch
import torch.nn as nn
import timm


class MediSwinGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True
        )

        self.up4 = nn.ConvTranspose2d(768, 384, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(384 + 384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(192 + 192, 192, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(96, 64, 2, stride=2)

        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):

        feats = self.encoder(x)

        f1, f2, f3, f4 = feats

        f1 = f1.permute(0,3,1,2)
        f2 = f2.permute(0,3,1,2)
        f3 = f3.permute(0,3,1,2)
        f4 = f4.permute(0,3,1,2)

        x = self.up4(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, f2], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.dec2(x)

        x = self.up1(x)

        x = self.up0(x)

        x = self.final(x)

        return x