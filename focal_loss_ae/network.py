import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):

    def _build_networks(self):
              h = tf.layers.conv2d(self.x, 32, 4, strides=2,
                                   activation=tf.nn.relu, name="enc_conv1")
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
        ) # 64 - 32 - 16 - 8 - 6 - 4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3), # 6
            nn.relu(),
            nn.ConvTranspose2d(64, 64, kernel_size=3), # 8
            nn.relu(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, dilation=2, stride=2, output_padding=1), # 16
            nn.relu(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, dilation=2, stride=2, output_padding=1), # 32
            nn.relu(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, dilation=2, stride=2, output_padding=1),
            nn.sigmoid()
        )
    



    def forward(self, x):
