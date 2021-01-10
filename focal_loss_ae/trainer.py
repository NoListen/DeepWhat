import torch
from torch import nn
from collections import namedtuple
from enum import IntEnum
from torch.utils.tensorboard import SummaryWriter

opt_params = namedtuple("opt_params", ["lr"])


class TrainerPhase(IntEnum):
    train = 0
    test = 1


class VAELoss(nn.Module):
    def __init__(self, kl_loss_threshold=0, use_focal_loss=False):
        self.kl_loss_threshold = torch.Tensor(kl_loss_threshold)
        self.use_focal_loss = use_focal_loss

    def forward(self, input, output, mu, logvar):
        z_size = mu.shape[1]
        # logistic regression.
        r_loss = -torch.sum(batch * torch.log(output+1e-8) +
                            (1-batch) * torch.log(1-output+1e-8), dim=[1, 2, 3])
        r_loss = torch.mean(r_loss, dim=0)

        kl_loss = -0.5 * \
            torch.sum(1+logvar-torch.square(mu)-torch.exp(logvar), axis=1)
        kl_loss = torch.max(kl_loss, self.kl_loss_threshold)
        kl_loss = torch.mean(kl_loss)

        loss = kl_loss + r_loss

        return loss


class AETrainer:
    def __init__(self, network, loss, opt_params):
        self.network = network
        self.loss = loss
        self.optimizer = None
        self.opt_params = opt_params
        self.phase = TrainerPhase.train
        self.train_step = 0
        self.writer = SummaryWriter()

    def set_phase(self, phase: TrainerPhase):
        self.phase = phase

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),  self.opt_params.lr)

    def train(self, batch):
        mu, logvar, output = self.network(batch)
        loss = self.loss(batch, output, mu, logvar)
        if self.phase == TrainerPhase.train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_step += 1
            self.writer.add_scalar("Loss/train", loss, self.train_step)
