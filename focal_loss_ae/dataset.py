import os
import os.path as osp
from typing import List

# This dataset aims at preloading the data from the disk.


class DiskDataset:
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

