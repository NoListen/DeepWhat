import torch
from torch import nn


class VAELoss(nn.Module):
    def __init__(self, kl_loss_threshold=0, use_focal_loss=False):
        self.kl_loss_threshold = kl_loss_threshold
        self.use_focal_loss = use_focal_loss

    def forward(self, input, output, mu, logvar):
        z_size = mu.shape[1]
        r_loss = -torch.sum(batch * torch.log(output+1e-8) +
                            (1-batch) * torch.log(1-output+1e-8), dim=[1, 2, 3])
        r_loss = torch.mean(r_loss, dim=0)

        kl_loss = -0.5 * \
            torch.sum(1+logvar-torch.square(mu)-torch.exp(logvar), axis=1)
        kl_loss = torch, maximum(kl_loss, self.kl_loss_threshold)
        kl_loss = torch.mean(kl_loss)

        loss = kl_loss + r_loss

        return loss


class AETrainer:
    def __init__(self, network, loss):
        self.network = network
        self.loss = loss
        self.optimizer = None

    def _build_optimizer(self):
        self.optimizer

    def train(self, batch):
        mu, logvar, output = self.network(batch)
        loss = self.loss(batch, output, mu, logvar)