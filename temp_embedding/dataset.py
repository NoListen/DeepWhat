import os
import os.path as osp
from typing import Tuple, List

from collections import defaultdict
import numpy as np
import pickle

def get_nested_target_file_paths(data_dir: str, relative_path: str="", suffix=".p"):
    sub_data_dir = osp.join(data_dir, relative_path)n 
    file_names = os.listdir(sub_data_dir)
    
    file_names = sorted(file_names)
    file_paths = []

    leave_node_flag = False
    
    for fn in file_names:
        if osp.isdir(osp.join(sub_data_dir, fn)):
            file_paths += get_target_file_paths(data_dir,
                                                osp.join(relative_path, fn), suffix)
        elif suffix in fn:
            file_paths.append(osp.join(relative_path, fn))
            leave_node_flag = True
    
    # Nested list
    if leave_node_flag:
        file_paths = [file_paths]

    return file_paths

def get_episode_name(file_path):
    return file_path.split('/')[-2]

def clip_interval_within_range(interval_start:int, interval_end: int,
                               start:int, end:int)->Tuple[int, int]:
    interval_start = np.clip(interval_start, start, end)
    interval_end = np.clip(interval_end, start, end)
    return (interval_start, interval_end)


def load_single_data(file_path):
    with open(file_path, "rb") as f:
            data = pickle.load(file_path)
    return data
    
class DiskDataset:
    def __init__(self, data_root, nested_file_names, neighbor_distance=5):
        self.data_root = data_root

        self.file_names = [f for ep_file_names in nested_file_names
                             for f in sorted(ep_file_names)[:-1]]
        self.next_file_names = [f for ep_file_names in nested_file_names
                                  for f in sorted(ep_file_names)[1:]]
        
        self.ep_file_names_dict = defaultdict(list)
        self.ep_start_id_dict = defaultdict(int)
        self.file_names = []
        self.next_file_names = []
        self.neighbor_distance = neighbor_distance

        self._init_file_names(nested_file_names)
    
    def _init_file_names(self, nested_file_names):        
        for ep_file_names in nested_file_names:
            ep_file_names = sorted(ep_file_names)

            self.file_names += ep_file_names[:-1]
            self.next_file_names += ep_file_names[1:]

            episode_name = get_episode_name(f)
            self.ep_file_names_dict[episode_name] = ep_file_names[:-1]
            self.ep_start_id_dict[episode_name] = file_index
            file_index += (len(ep_file_names) - 1)
        
        assert file_index == len(sel.file_names), "the start id is not correct"


    def set_neighbor_distance(self, neighbor_distance:int):
        self.neighbor_distance = neighbor_distance

    def __len__(self):
        return len(self.file_names)

    # Get the nearest neighbors
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.load_data(index)
    

    def load_data(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        anchor, action, after_anchor = self._load_single_obs_action_next_obs(index)
        pos_index, neg_index = self._get_pos_neg_indices(index, self.neighbor_distance)
        pos = self._load_single_obs(pos_index)
        neg = self._load_single_obs(neg_index)
        return {"anchor":anchor, "action": action, "after_anchor": after_anchor, "pos":pos, "neg":neg}

    def _load_single_obs(self, index:int):
        file_path = os.path.join(self.data_root, self.file_names[index])
        return load_single_data(file_path)["obs"][None]
    
    def _load_single_obs_action_next_obs(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        file_path = os.path.join(self.data_root, self.file_names[index])
        next_file_path = os.path.join(self.data_root, self.next_file_names[index])

        data = load_single_data(file_path)
        next_data = load_single_data(next_file_path)

        return data["obs"][None], np.array(data["action"])[None], next_data["obs"][None]

        
    def _load_single_obs(self, index: int) -> np.ndarray:
        file_path = os.path.join(self.data_root, self.file_names[index])
        return np.load(file_path)[None].astype(np.float32)
        
    # this function would be super important for sampling data.
    def _get_pos_neg_indices(self, index, neighbor_distance=5) -> Tuple[int, int]:
        episode_name = get_episode_name(self.file_names[index])
        episode_start_id = self.ep_start_id_dict[episode_name]
        episode_file_names = self.ep_file_names_dict[episode_name]

        episode_id = index - episode_start_id
        episode_length = len(episode_file_names)
        if episode_length < 2*neighbor_distance+1:
            raise ValueError(f"the episode lengh is too small to sample non-neighbor samples")
        neighbor_start, neighbor_end = clip_interval_within_range(episode_id-neighbor_distance,
                                                                  episode_id+neighbor_distance+1,
                                                                  0,
                                                                  episode_length)
        neighbor_index = np.random.randint(neighbor_start, neighbor_end)
        non_neighbor_index = np.random.randint(0, episode_length-(neighbor_end-neighbor_start))
        if non_neighbor_index >= neighbor_start:
            non_neighbor_index = non_neighbor_index - neighbor_start + neighbor_end
        return neighbor_index+episode_start_id, non_neighbor_index+episode_start_id


        

