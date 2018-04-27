import numpy as np
from math import sqrt, exp
from copy import deepcopy
import matplotlib.pyplot as plt
import random

# n means the number of sites
def generate_data(n, domain=(0,100)):
    low, high = domain
    points = np.random.uniform(low, high, (n, 2)) # x,y 2 X N
    return points

def generate_cluster_data(N, maxn=5, std=3, domain=(0, 100)):
    # each cluster have at least one site
    # N clusters

    low, high = domain
    cpoints = np.random.uniform(low, high, (N, 2))
    points = []
    for i in range(N):
        n = np.random.randint(max(1, int(maxn*0.2)), maxn+1)
        data_noise = np.random.randn(n, 2)*std
        points_incre = cpoints[i] + data_noise
        points.append(points_incre)
    points = np.concatenate(points, axis=0)
    return points

def calculate_distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_distances(points):
    """
    :param points: (x, y) numpy array
    :return: average distance
    """
    # they must be equal
    n = len(points)
    if n == 1:
        raise Exception("No need to compute the distances")
    distances = np.zeros((n, n))
    # sum = 0
    for i in range(n):
        distances[i][i] = np.inf
        for j in range(i+1, n):
            distances[j][i] = distances[i][j] = calculate_distance(points[i], points[j])
            # sum += distances[i][j]
    return distances

def get_cost(l1, l2, distances):
    return distances[l1]+distances[l2]

def get_mean_std(distances):
    n = distances.shape[0]
    true_distances = [distances[i][j] for i in range(n) for j in range(i+1,n)]
    mean = np.mean(true_distances)
    std = np.std(true_distances)
    return mean, std

def get_cost_table(distances, mean, scale):
    ctable = (distances-mean)/scale
    return ctable

def get_potential_table(ctable):
    return np.exp(-ctable)


def get_path_length(distances, path):
    length = 0
    N = len(path)
    for i in range(N):
        length += distances[path[i], path[(i+1)%N]]
    return length


# randomly select n indices out of l
def random_sample_indices(l, n):
    return np.random.choice(l, n, replace=False)

#TODO select one subset and swap. by clustering
def soft_random_select_swap(N, distances, initial_path, max_iters=int(1e6)):
    """
    :param N: the number of data points
    :param maxn: the maximum number of swapping subset
    :return:
    """
    accumulated_cost = 0
    min_cost = 0
    best_path = initial_path
    path = initial_path
    print("initial path length", get_path_length(distances, path))
    t = 100
    mean, std = get_mean_std(distances)
    all_indices = np.arange(N)
    for i in range(max_iters):


        if (i+1)%200:
            t = max(t*0.9995, 0.1)

        sample_indices = random_sample_indices(all_indices, 2)
        next_indices = (sample_indices+1)%N
        s1, s2 = path[sample_indices]
        n1, n2 = path[next_indices]
        # print(path, s1, n1, s2, n2)
        if s2 == n1:
            continue

        pre_cost = get_cost((s1, n1), (s2, n2), distances)
        cost = get_cost((s1, s2), (n1, n2), distances)

        diff = cost - pre_cost
        if diff < 0:
            prob = 1
        else:
            prob = exp(-diff/t)

        # swap their neighbors
        if random.uniform(0, 1) < prob:
            accumulated_cost += diff
            l_reverted = (N + sample_indices[1] - next_indices[0])%N + 1
            indices_reverted = [id%N for id in range(next_indices[0], next_indices[0]+l_reverted)]
            path[indices_reverted] = path[indices_reverted][::-1]
            if accumulated_cost < min_cost:
                best_path = deepcopy(path)
                min_cost = accumulated_cost
    return best_path

def plot_data(points):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1])
    # plt.show()

def plot_path(points, order, show=False):
    plt.figure()
    N = len(order)
    for i in range(len(order)):
        p1 = order[i]
        p2 = order[(i+1)%N]
        plt.plot([points[p1,0], points[p2,0]], [points[p1,1], points[p2,1]], 'ro-')
    if show:
        plt.show()
