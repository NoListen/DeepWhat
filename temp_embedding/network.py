import torch
import torch.nn as nn

BN_MOMENTUM = 0.1


# TODO(lisheng) Apply them to randomly collected data and expert experience.
# Input size is expected to 64.
class TempEmbed(nn.Module):
    def __init__(self):
        super(TempEmbed, self).__init__()
        self.temp_embed = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2), # 32
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 16
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2), # 8
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # 6,
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Linear(2304, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Sigmoid()
        )
    
    def forward(self, batch):
        return self.temp_embed(batch)