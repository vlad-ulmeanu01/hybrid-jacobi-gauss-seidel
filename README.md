# Abstract

Let us consider Gauss-Seidel’s method for solving systems of linear equations through a sequence of iterations. Compared
with the matriceal variant, the analytic variant is expressed in such a way that it cannot be simply parallelized, requiring additional data
that didn’t exist in any previous step in any particular moment. The suggested method proposes that each new variable should lose
some amount of new data in the recurrence formula, such that the solution vector could be split into buckets, resulting in any bucket’s
contents being calculated in parallel.

We present numerical results for strictly diagonally dominant matrices, for which the runtime does significantly improve, and the
required number of iterations drops as well.

Furthermore, an analytic heuristic approach for Gauss-Seidel is shown that may shuffle the rows of the solution vector during each
step, with an O(n<sup>2</sup> + n log n) complexity per iteration. Its convergence rate is on par with SOR, and it permits parallel implementations.

# Index Terms

Gauss-Seidel, Parallel Computing, Hybrid Methods, Optimization, Dynamic Programming.
