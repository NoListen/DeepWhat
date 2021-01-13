from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
import os
import torch
import argparse
from dataset import DiskDataset, get_target_file_paths
from network import AutoEncoder
from trainer import AETrainer, VAELoss, opt_params
from torch.utils.tensorboard import SummaryWriter


def train(data_dir, lr, batch_size, num_workers, num_epochs, z_size,
          kl_tolerance, log_dir, use_focal_loss=False, use_gpu=False):
    file_paths = get_target_file_paths(data_dir)
    print(f"Get {len(file_paths)} images")
    dataset = DiskDataset(data_dir, file_paths)
    # for i in tqdm(range(len(dataset))):
    # dataset[i]
    if use_gpu and torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)

    ae = AutoEncoder(z_size).to(dev)
    kl_loss_threshold = kl_tolerance * z_size
    loss = VAELoss(kl_loss_threshold,
                   use_focal_loss=use_focal_loss, dev=dev).to(dev)
    print("the device is", dev)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    summary_writer = SummaryWriter(log_dir=log_dir)
    ae_opt_params = opt_params(lr=lr)
    trainer = AETrainer(ae, loss, ae_opt_params, summary_writer)

    num_batches = len(file_paths)//batch_size
    for i in range(num_epochs):
        print(f"epoch {i}")
        for _ in tqdm(range(num_batches)):
            data = next(iter(dataloader))
            data = data.to(dev)
            trainer.train(data)
        torch.save(ae.state_dict(), os.path.join(log_dir, f"model{i}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train variational autoencoder on the the train dataset.")

    parser.add_argument("--data-dir", type=str,
                        help="The data directory to load data recursively")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="the learning rate of the model")
    parser.add_argument("--batch-size", type=int,
                        default=16, help="the batch_size for training")
    parser.add_argument("--num-workers", type=int,
                        default=8, help="the number of dataloader workers")
    parser.add_argument("--num-epochs", type=int,
                        default=10, help="the number of training epochs")
    parser.add_argument("--z-size", type=int,
                        default=64, help="the size of hidden vector")
    parser.add_argument("--kl-tolerance", type=float,
                        default=0.5, help="the kl tolerance ratio regard to the z size")
    parser.add_argument("--use-focal-loss",
                        action="store_true", help="use focal loss")
    parser.add_argument("--log-dir", type=str, default="./exp/log",
                        help="the log directory to log the information")
    parser.add_argument("--use-gpu", action="store_true",
                        help="use gpu or not")
    args = parser.parse_args()
    args = vars(args)
    print(args)
    train(**args)
