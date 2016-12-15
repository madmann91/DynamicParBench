# DynamicParBench

This project is a benchmark to test the effectiveness of 4 different parallelization techniques for a simple problem. The problem is as follows:

_Given two arrays_ `N` _and_ `D` _containing integers, compute the array made of:_
    
    D(0) + 0, D(0) + 1, ..., D(0) + N(0) - 1, D(1) + 0, D(1) + 1, ... , D(1) + N(1) - 1, ..., D(m) + N(m) - 1
        
The number of elements in the array `N` determines how many times a specific element of `D` will be used. The algorithm produces `N(0) + N(1) + N(2) + ... + N(m)` values as output, and the distribution of the values in `N` can be either regular (`N(i) = constant`) or completely irregular.

The 4 strategies are:

- Naive approach (each thread i has to generate N(i) elements)
- Dynamic parallelism (launching a child kernel whenever `N(i)` is too high)
- Binary search (each thread writes one value, and the corresponding `D(i)` is found using binary search)
- Blocking (switches the loops to maximize the number of active threads)

The benchmarks indicate that the winning strategy is blocking: It is extremely efficient for both regular and completely irregular workloads.
