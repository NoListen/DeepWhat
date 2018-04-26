import numpy as np
import random
from utils import *
np.random.seed(1234567)
random.seed(1234567)

Shuffle = False
points = generate_data(100)
# points = generate_cluster_data(20, maxn=5, std=3, domain=(0, 100))
n = len(points)
print("We have gotten %i sites to travel" % n)

# those in the same cluster may be close to each other in the indices
# because the order of processing
# To check the initialization's effect
# you can change Shuffle's values
if Shuffle:
    order = np.arange(n)
    np.random.shuffle(order)
    points = points[order]

distances = get_distances(points)
# mean, std = get_mean_std(distances)
plot_data(points)

# Method 1 randomly swapping
# a rough loop
initial_path = np.arange(n)

path = initial_path
plot_path(points ,path, show=False)

path = soft_random_select_swap(n, distances, initial_path, max_iters=int(1e6))
print(path)
plot_path(points, path, show=True)
