import numpy as np
from math import sqrt
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

def get_mean_std(distances):
    n = distances.shape[0]
    true_distances = [distances[i][j] for i in range(n) for j in range(i+1,n)]
    mean = np.mean(true_distances)
    std = np.std(true_distances)
    return mean, std

def get_cost_potential_table(distances, mean, scale):
    ctable = (distances-mean)/scale
    ptable = np.exp(-ctable)
    return ctable, ptable

def get_cost(distances, sample_indices, sample_neighbors):
    cost = 0
    for i in range(len(sample_indices)):
        left = sample_neighbors[i, 0]
        right = sample_neighbors[i, 1]
        cost += distances[sample_indices[i], left]
        cost += distances[sample_indices[i], right]
    return cost

# form the aspects of prob
def get_cost_potential(ctable, ptable, sample_indices, sample_neighbors):
    cost = 0
    potential = 1
    for i in range(len(sample_indices)):
        id = sample_indices[i]
        left = sample_neighbors[i, 0]
        right = sample_neighbors[i, 1]
        cost += (ctable[id, left] + ctable[id, right])
        potential *= (ptable[id, left] * ptable[id, right])
    return cost, potential

def get_neighbors(path):
    N = len(path)
    neighours = np.zeros((N, 2), dtype=np.int32)
    for i in range(N):
        neighours[i, 0] = path[(i+N-1)%N]
        neighours[i, 1] = path[(i+1)%N]
    return neighours

def process_neighbors(sample_indices, sample_neighbors):
    p1, p2 = sample_indices
    if p1 in sample_neighbors[1]: # they are neighbors
        if sample_neighbors[1, 0] == p1:# n1 <-> p1 <-> p2 <-> n2 |=> n1 <-> p2 <-> p1 <-> n2
            sample_neighbors[0, 1] = p1
            sample_neighbors[1, 0] = p2
        else:                           # n1 <-> p2 <-> p1 <-> n2 |=> n1 <-> p1 <-> p2 <-> n2
            sample_neighbors[1, 0] = p1
            sample_neighbors[0, 1] = p2
    return sample_neighbors

def construct_path_from_neighbors(neighbors):
    N = len(neighbors)
    p = neighbors[0, 1]
    path = [p]
    # print("Calculating Path")
    # print(neighbors)
    while p!=0:
        p = neighbors[p, 1]
        path.append(p)
    # print(len(path), N)
    if len(path) != N:
        raise Exception("Invalid path")
    return path


# randomly select n indices out of l
def random_sample_indices(l, n):
    return np.random.choice(l, n, replace=False)


#TODO select one subset and swap.
# Greedy swapping
def random_select_swap(N, distances, initial_path, max_iters=int(1e5)):
    """
    :param N: the number of data points
    :param maxn: the maximum number of swapping subset
    :return:
    """
    neighbors = get_neighbors(initial_path)
    all_indices = np.arange(N)
    for i in range(max_iters):
        sample_indices = random_sample_indices(all_indices, 2)
        sample_neighbors = neighbors[sample_indices]
        previous_cost = get_cost(distances, sample_indices, sample_neighbors)

        sample_neighbors = process_neighbors(sample_indices, sample_neighbors)
        cost = get_cost(distances, sample_indices, sample_neighbors[::-1])
        # swap their neighbors
        if cost < previous_cost:
            print(previous_cost, cost, "in iteration %i" % i)
            neighbors[sample_indices[0]] = sample_neighbors[1]
            neighbors[sample_indices[1]] = sample_neighbors[0]
            neighbors[sample_neighbors[0, 0], 1] = sample_indices[1]
            neighbors[sample_neighbors[0, 1], 0] = sample_indices[1]
            neighbors[sample_neighbors[1, 0], 1] = sample_indices[0]
            neighbors[sample_neighbors[1, 1], 0] = sample_indices[0]
            # print(neighbors)
    print("end")
    path = construct_path_from_neighbors(neighbors)
    return path

#TODO select one subset and swap. by clustering
def soft_random_select_swap(N, distances, initial_path, max_iters=int(1e5)):
    """
    :param N: the number of data points
    :param maxn: the maximum number of swapping subset
    :return:
    """
    neighbors = get_neighbors(initial_path)
    accumulated_cost = 0
    min_cost = 0
    best_path = initial_path
    mean, std = get_mean_std(distances)
    print("the stand deviation is", std)
    ctable, ptable = get_cost_potential_table(distances, mean, std/10.)

    all_indices = np.arange(N)
    for i in range(max_iters):
        sample_indices = random_sample_indices(all_indices, 2)
        sample_neighbors = neighbors[sample_indices]
        pre_cost, pre_potential = get_cost_potential(ctable, ptable, sample_indices, sample_neighbors)

        sample_neighbors = process_neighbors(sample_indices, sample_neighbors)
        cost, potential = get_cost_potential(ctable, ptable, sample_indices, sample_neighbors[::-1])
        # the probability of swapping
        prob = potential/(potential+pre_potential)
        # swap their neighbors
        if random.uniform(0, 1) < prob:
            accumulated_cost += (cost-pre_cost)
            if accumulated_cost < min_cost:
                best_path = construct_path_from_neighbors(neighbors)
                min_cost = accumulated_cost
            neighbors[sample_indices[0]] = sample_neighbors[1]
            neighbors[sample_indices[1]] = sample_neighbors[0]
            neighbors[sample_neighbors[0, 0], 1] = sample_indices[1]
            neighbors[sample_neighbors[0, 1], 0] = sample_indices[1]
            neighbors[sample_neighbors[1, 0], 1] = sample_indices[0]
            neighbors[sample_neighbors[1, 1], 0] = sample_indices[0]
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
