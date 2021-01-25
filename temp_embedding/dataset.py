import os
from typing import Tuple

from collections import defaultdict
import numpy as np


def get_episode_name(file_path):
    return file_path.split('/')[-2]

def clip_interval_within_range(interval_start:int, interval_end: int,
                               start:int, end:int)->Tuple[int, int]:
    interval_start = np.clip(start, end)
    interval_end = np.clip(start, end)
    return (interval_start, interval_end)
    

class DiskDataset:
    def __init__(self, data_root, file_names, neighbor_distance=5):
        self.data_root = data_root
        self.file_names = file_names
        self.ep_file_names_dict = defaultdict(lost)
        self.ep_start_id_dict = defaultdict(int)
        
        for i,f in enumerate(file_names):
            episode_name = get_episode_name(f)
            self.ep_file_names_dict[episode_name] = f
            if episode_name not in self.ep_start_id_dict:
                self.ep_start_id_dict[episode_name] = i
        self.neighbor_distance = neighbor_distance

    def set_neighbor_distance(self, neighbor_distance:int):
        self.neighbor_distance = neighbor_distance

    def __len__(self):
        return len(self.file_names)

    # Get the nearest neighbors
    def __get_item__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.load_data(index)
    

    def load_data(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        anchor = self._load_single_data(index)
        pos_index, neg_index = self._get_pos_neg_indices(index, self.neighbor_distance)
        pos = self._load_single_data(pos_index)
        neg = self._load_single_data(neg_index)
        return {"anchor":anchor, "pos":pos, "neg":neg}
    
    def _load_single_data(self, index: int) -> np.ndarray:
        file_path = os.path.join(data_root, self.file_names[index])
        return np.load(file_path)
        
    # this function would be super important for sampling data.
    def _get_pos_neg_indices(self, index, neighbor_distance=5) -> Tuple[int, int]:
        episode_name = get_episode_name(index)
        episode_start_id = self.ep_start_id_dict[episode_name]
        episode_file_names = self.ep_file_names[episode_name]

        episode_id = index - episode_start_id
        episode_length = len(episode_file_names)
        if episode_length < 2*neighbor_distance+1:
            raise ValueError(f"the episode lengh is too small to sample non-neighbor samples")
        neighbor_start, neighbor_end = clip_interval_within_range(episode_id-neighbor_distance,
                                                                  episode_id+neightbor_distance+1,
                                                                  0,
                                                                  episode_length)
        neighbor_index = np.random.randint(neighbor_start, neighbor_end)
        non_neighbor_index = np.random.randint(0, episode_length-len(neighbor_end-neighbor_start))
        if non_neighbor_index >= neighbor_start:
            non_neighbor_index = non_neighbor_index - neighbor_start + neighbor_end
        ruturn neighbor_index+episode_start_id, non_neighbor_index+episode_start_id


        

