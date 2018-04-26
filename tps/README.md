# Introduction

Use MCMC method to find a pretty path in TPS problem.

# Done

Randomly select a pair of nodes.
- Swap: If the cost after swap is lower than previous one, than swap the two points' neighbors.
- Soft Swap: Calculate potential funcation `f=e^{g(l)}` where l is the length of them. Then calculate the probability of swapping and randomly choose to do so or not.

# Process
**Randomly generate 100 points in the plane**


![Data Generation](data/points.png?raw=true)

**Initialized a random path**


![Random Path/Loop](data/tps_init.png?raw=true)

**After 1e6 MCMC iterations**


![Result Path/Lopp](data/tps_100.png?raw=true)


It's promising to adjust the mean and scale to find better path in `soft_random_select_swap`. These parameters determine how you care about the distances in neighbors exchange.

# TODO
- Restore the previous best path if the search algorithm can't find better one in a long time which may be caused by going too far.
- Make use of clusterring
