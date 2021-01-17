import torch
import torch.nn as nn

# TODO(lisheng) Would cosine similarity be better.
class TempTrainer(nn.Module):
    def __init__(self, model, lr=1e-3, margin=0.1, p=2):
        super(TripletLoss, self).__init__()
        self.loss = nn.TripletMarginLoss(margin, p)
        self.model = model
        self.optimizer = None
        self._build_optimizer(lr)
        self.train_step += 1

    def _build_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    
    def train(self, batch):
        anchor = self.model(batch["anchor"])
        postive = self.model(batch["positive"])
        negative = self.model(batch["negative"])
        triplet_loss = self.loss(anchor, postive, negative)

        self.optimizer.zero_grad()
        triplet_loss.backward()
        self.optimizer.step()
        self.train_step += 1
