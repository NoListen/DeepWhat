import torch
import torch.nn as nn

# TODO(lisheng) Would cosine similarity be better.
class TempTrainer(nn.Module):
    def __init__(self, model, writer, lr=1e-3, margin=0.1, p=2):
        super(TempTrainer, self).__init__()
        self.loss = nn.TripletMarginLoss(margin, p)
        self.model = model
        self.optimizer = None
        self.writer = writer
        self._build_optimizer(lr)
        self.train_step = 0

    def _build_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    
    def train(self, batch):
        anchor = self.model(batch["anchor"])
        postive = self.model(batch["pos"])
        negative = self.model(batch["neg"])
        triplet_loss = self.loss(anchor, postive, negative)

        self.optimizer.zero_grad()
        triplet_loss.backward()
        self.optimizer.step()
        self.train_step += 1
        self.writer.add_scalar("Loss/train", triplet_loss, self.train_step)
