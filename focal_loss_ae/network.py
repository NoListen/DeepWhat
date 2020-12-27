import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, z_size):
        self.encoder = None
        self.decoder = None
        self.z_size = z_size
        self._build_networks()

    def _build_networks(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, dilation=2, stride=2, padding=2),
            nn.relu(),
            nn.Conv2d(32, 64, kernel_size=3, dilation=2, stride=2, padding=2),
            nn.relu(),
            nn.Conv2d(64, 64, kernel_size=3, dilation=2, stride=2, padding=2),
            nn.relu(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.relu(),
            nn.Conv2d(64, 64, kernel_size=3)
        )  # 64 - 32 - 16 - 8 - 6 - 4

        self.mu = nn.Linear(1024, self.z_size)
        self.logvar = nn.Linear(1024, self.z_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3),  # 6
            nn.relu(),
            nn.ConvTranspose2d(64, 64, kernel_size=3),  # 8
            nn.relu(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, dilation=2,
                               stride=2, output_padding=1),  # 16
            nn.relu(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, dilation=2,
                               stride=2, output_padding=1),  # 32
            nn.relu(),
            nn.ConvTranspose2d(32, 3, kernel_size=3,
                               dilation=2, stride=2, output_padding=1),
            nn.sigmoid()
        )

    def forward(self, x):
        e = self.encoder(x)
        print("the input after the encoder network", e.shape)
        e = torch.reshape(e, (-1, 1024))
        mu = self.mu(e)
        logvar = self.logvar(e)
        sigma = torch.exp(logvar/2.)
        z = mu + torch.randn_like(sigma) * sigma
        output = self.decoder(z)
        return output
