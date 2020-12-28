import os
import torch
import os.path as osp
from typing import List


def get_target_file_paths(data_dir: str, relative_path: str = "", suffix=".npy"):
    # List all paths relative to the data dir.
    file_names = os.path.listdir(data_dir)
    file_paths = []
    for fn in file_names:
        if os.isdir(fn):
            file_paths += get_target_file_paths(data_dir,
                                                osp.join(relative_path, fn), suffix)
        elif suffix in fn:
            file_paths.append(osp.join(relative_path, fn))
    return file_paths


class DiskDataset(torch.utils.data.Dataset):
   def __init__(self, data_dir: str, filenames: List[str]):
       self.filenames = filenames
       self.data_dir = data_dir
       self.length = len(filenames)

    def __getitem__(self, idx):
        self.load_data(idx)

    def load_data(self, idx: int) -> dict:
        file_path = osp.join(self.data_dir, self.filenames[idx])
        data = np.load(file_path)
        return {"input": data}

    def __len__(self) -> int:
        return self._length

