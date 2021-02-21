import torch
import torch.nn as nn

BN_MOMENTUM = 0.1


# Input size is expected to 64.
class TempEmbed(nn.Module):
    def __init__(self):
        super(TempEmbed, self).__init__()
        self.temp_embed_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2), # 32
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 16
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2), # 8
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 6
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.temp_embed_fc = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Sigmoid()
        )
    
    def forward(self, batch):
        conv_embed = self.temp_embed_conv(batch)
        fc_embed = self.temp_embed_fc(conv_embed.view(batch.shape[0], 2304))
        return fc_embed

class InverseDynamics(nn.Module):
    def __init__(self, na):
        super(InverseDynamics, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, na),
            nn.Softmax(dim=1)
        )
    
    def forward(self, embed_a, embed_b):
        embed = torch.cat((embed_a, embed_b), axis=1)
        action_logits = self.model(embed)
        
        return action_logits
        