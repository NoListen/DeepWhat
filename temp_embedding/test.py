from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
import os
import torch
import argparse
import numpy as np
from dataset import DiskDataset, get_target_file_paths
from network import TempEmbed
from trainer import TempTrainer
from torch.utils.tensorboard import SummaryWriter


def train(data_dir, model_path, slice, use_gpu=False):
    file_paths = get_target_file_paths(data_dir)
    print(f"Get {len(file_paths)} images")
    dataset = DiskDataset(data_dir, file_paths)
    if use_gpu and torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    state_dict = torch.load(model_path)
    model = TempEmbed().to(dev)
    model.load_state_dict(state_dict)
    print("the device is", dev)
    embeddings = []
    for i in range(slice):
        data = dataset._load_single_data(i)[None]

        data = torch.from_numpy(data).to(dev)
        embed = model(data)
        embeddings.append(embed.detach().cpu().numpy())

    distances = np.zeros((slice, slice), dtype=np.float32)
    for i in range(slice):
        for j in range(i+1, slice):
            difference_square = (embeddings[i] - embeddings[j])**2
            distances[i][j] = np.sqrt(difference_square.sum())
    
    np.save("distances2.npy", distances)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train variational autoencoder on the the train dataset.")

    parser.add_argument("--data-dir", type=str,
                        help="The data directory to load data recursively")
    parser.add_argument("--model-path", type=str,
                        help="the path to load the model")
    parser.add_argument("--slice", type=int, help="the slice of dataset to test")
    parser.add_argument("--use-gpu", action="store_true",
                        help="use gpu or not")
    args = parser.parse_args()
    args = vars(args)
    print(args)
    train(**args)
