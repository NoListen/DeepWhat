from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
import os
import torch
import argparse
import numpy as np
from dataset import DiskDataset, get_nested_target_file_paths
from network import TempEmbed, InverseDynamics
from trainer import TempTrainer
from torch.utils.tensorboard import SummaryWriter


def train(data_dir, lr, batch_size, num_workers, num_epochs, neighbor_distance, log_dir,
          use_focal_loss=False, use_gpu=False):
    # number of discrete actions.
    na = 18
    
    file_paths = get_nested_target_file_paths(data_dir)
    print(f"Get {len(file_paths)} episodes")
    dataset = DiskDataset(data_dir, file_paths, neighbor_distance)
    print(f"Get {len(dataset)} entries")
    if use_gpu and torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)

    model = TempEmbed().to(dev)
    inverse_dynamics_model = InverseDynamics(na=na).to(dev)

    print("the device is", dev)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    summary_writer = SummaryWriter(log_dir=log_dir)
    trainer = TempTrainer(model, inverse_dynamics_model, summary_writer, lr)

    num_batches = len(dataset)//batch_size
    for i in range(num_epochs):
        print(f"epoch {i}")
        for _ in tqdm(range(num_batches)):
            data = next(iter(dataloader))
            data = {k: v.to(dev) for k, v in data.items()}
            trainer.train(data)
        torch.save(model.state_dict(), os.path.join(log_dir, f"model{i}.pth"))
        torch.save(inverse_dynamics_model.state_dict(), os.path.join(log_dir, f"id_model{i}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train variational autoencoder on the the train dataset.")

    parser.add_argument("--data-dir", type=str,
                        help="The data directory to load data recursively")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="the learning rate of the model")
    parser.add_argument("--batch-size", type=int,
                        default=32, help="the batch_size for training")
    parser.add_argument("--num-workers", type=int,
                        default=8, help="the number of dataloader workers")
    parser.add_argument("--num-epochs", type=int,
                        default=10, help="the number of training epochs")
    parser.add_argument("--log-dir", type=str, default="./exp/log",
                        help="the log directory to log the information")
    parser.add_argument("--use-gpu", action="store_true",
                        help="use gpu or not")
    parser.add_argument("--neighbor-distance", type=int, default=5, help="the range to select neighbors")
    args = parser.parse_args()
    args = vars(args)
    print(args)
    train(**args)
