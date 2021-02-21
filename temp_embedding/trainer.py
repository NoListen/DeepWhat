import torch
import torch.nn as nn

# TODO(lisheng) Would cosine similarity be better.
class TempTrainer(nn.Module):
    def __init__(self, model, inverse_dynamics_model, writer, lr=1e-3, margin=0.1, p=2):
        """
        model:  the temp embedding model to train.
        inverse_dynamics_model: the inverse dynamics model.
        writer: the tensorboard summary writer.
        lr:     the learning rate of the optimizer.
        margin: the margin to be used in TripletMarginLoss.
        p:      the power for the loss measurement in TripletMarginLoss.
        """
        super(TempTrainer, self).__init__()
        self.model = model
        self.inverse_dynamics_model = inverse_dynamics_model

        self.loss = nn.TripletMarginLoss(margin, p)
        self.inverse_dynamics_loss = nn.CrossEntropyLoss()
        
        self.optimizer = None
        self.writer = writer
        self._build_optimizer(lr)
        self.train_step = 0

    def _build_optimizer(self, lr):
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        all_model_parameters = list(self.model.parameters()) + list(self.inverse_dynamics_model.parameters())
        self.optimizer = torch.optim.Adam(all_model_parameters, lr)
    
    # TODO(lisheng) Balance the loss.
    def train(self, batch):
        anchor = self.model(batch["anchor"])
        after_anchor = self.model(batch["after_anchor"])
        action_logits = self.inverse_dynamics_model(anchor, after_anchor)
        postive = self.model(batch["pos"])
        negative = self.model(batch["neg"])
        
        triplet_loss = self.loss(anchor, postive, negative)
        inverse_dynamics_loss = self.inverse_dynamics_loss(action_logits, batch["action"])

        self.optimizer.zero_grad()

        loss = triplet_loss + inverse_dynamics_loss
        loss.backward()
        self.optimizer.step()
        self.train_step += 1
        self.writer.add_scalar("Loss/train", triplet_loss, self.train_step)
