import torch
import argparse
from dataset import DiskDataset, get_target_file_paths
from network import AutoEncoder


def train(data_dir, batch_size, num_workers, num_epochs, z_size):
    file_paths = get_target_file_paths(data_dir)
    dataset = DiskDataset(data_dir, file_paths)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    ae = AutoEncoder(z_size)

    for i in range(num_epochs):
        for data in dataloader:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train variational autoencoder on the the train dataset.")

    parser.add_argument("--data-dir", required=True,
                        help="The data directory to load data recursively")
    parser.add_argument("--batch-size", type=int,
                        default=16, help="the batch_size for training")
    parser.add_argument("--num-workers", type=int,
                        default=8, help"the number of dataloader workers")
    parser.add_argument("--num-epochs", type=int,
                        default=10, help="the number of training epochs")
    parser.add_argument("--z-size", type=int,
                        default=64, help="the size of hidden vector")

    args = parser.parse_args()
    args = vars(args)
    train(*args)
