# Introduction

Use MCMC method to find a pretty path in the [Travelling Salesperson Problem(TSP)](https://en.wikipedia.org/wiki/Travelling_salesman_problem) problem.

# Done

Randomly select a pair of nodes.
- Swap: If the cost after swap is lower than previous one, than swap the two points' neighbors.
- Soft Swap: Calculate potential funcation `f=e^{g(l)}` where l is the length of them. Then calculate the probability of swapping and randomly choose to do so or not.

# Process
**Randomly generate 100 points in the plane**


![Data Generation](data/points.png?raw=true)

**Initialized a random path**


![Random Path/Loop](data/tsp_init.png?raw=true)

**After 1e6 MCMC iterations**


![Result Path/Lopp](data/tsp_100.png?raw=true)



# TODO
- Restore the previous best path if the search algorithm can't find better one in a long time which may be caused by going too far.
- Make use of clusterring
