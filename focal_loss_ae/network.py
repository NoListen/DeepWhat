import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, z_size):
        super(AutoEncoder, self).__init__()
        self.encoder = None
        self.decoder = None
        self.z_size = z_size
        self._build_networks()

    def _build_networks(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, dilation=2, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, dilation=2, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=2, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3)
        )  # 64 - 32 - 16 - 8 - 6 - 4

        self.mu = nn.Linear(1024, self.z_size)
        self.logvar = nn.Linear(1024, self.z_size)
        self.e2d = nn.Sequential(
            nn.Linear(self.z_size, 1024), nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3),  # 6
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3),  # 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, dilation=2,
                               stride=2, padding=2, output_padding=1),  # 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, dilation=2,
                               stride=2, padding=2, output_padding=1),  # 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3,
                               dilation=2, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, test=False):
        e = self.encoder(x)
        e = torch.reshape(e, (-1, 1024))
        mu = self.mu(e)
        logvar = self.logvar(e)
        sigma = torch.exp(logvar/2.)
        z = mu + torch.randn_like(sigma) * sigma
        dz = self.e2d(z).view(-1, 64, 4, 4)
        output = self.decoder(dz)

        if test:
            return output
        return mu, logvar, output
