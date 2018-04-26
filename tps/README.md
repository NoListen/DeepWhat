# Introduction

Use MCMC method to find a pretty path in TPS problem.

# Done

Randomly select a pair of nodes.
- Swap: If the cost after swap is lower than previous one, than swap the two points' neighbors.
- Soft Swap: Calculate potential funcation `f=e^{g(l)}` where l is the length of them. Then calculate the probability of swapping and randomly choose to do so or not.

# TODO
- Restore the previous best path if the search algorithm can't find better one in a long time which may be caused by going too far.
- Make use of clusterring
