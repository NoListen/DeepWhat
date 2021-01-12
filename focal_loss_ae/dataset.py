import os
import os.path as osp
from typing import List

import torch
import numpy as np

def get_target_file_paths(data_dir: str, relative_path: str = "", suffix=".npy"):
    # List all paths relative to the data dir.
    sub_data_dir = os.path.join(data_dir, relative_path)

    file_names = os.listdir(sub_data_dir)
    file_paths = []
    for fn in file_names:
        if os.path.isdir(osp.join(sub_data_dir, fn)):
            file_paths += get_target_file_paths(data_dir,
                                                osp.join(relative_path, fn), suffix)
        elif suffix in fn:
            file_paths.append(osp.join(relative_path, fn))
    return file_paths


class DiskDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, filenames: List[str]):
       self.filenames = filenames
       self.data_dir = data_dir
       self._length = len(filenames)
       print(self._length)
    
    def __getitem__(self, idx):
        return self.load_data(idx)

    def load_data(self, idx: int):
        file_path = osp.join(self.data_dir, self.filenames[idx])
        data = np.load(file_path)
        return data[None].astype(np.float32)

    def __len__(self) -> int:
        return self._length

