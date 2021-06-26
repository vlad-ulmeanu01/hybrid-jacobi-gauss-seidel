#include "utilities.h"
#include "jacobi.h"
#include "gauss_seidel.h"
#include "sor.h"
#include "hybrid.h"
#include "entropy.h"
#include "entropy_dp.h"
#include "tests.h"

/// problema de la -march=native
/// g++ -std=c++11 -O3 -fopenmp -march=native -lpthread -I ~/Eigen/ main.cpp -o main
/// g++ -std=c++17 -O3 -fopenmp -march=native -lpthread -I /usr/local/include/Eigen/ main.cpp -o main

///Explanation: Gauss-Seidel method is applicable to strictly diagonally dominant or
///symmetric positive definite matrices because only in this case convergence is possible.

int
main()
{
  Eigen::initParallel();
  ///nu ajuta pt versiunea asta Eigen, dar ar tb pus pt ca folosesc si eu op mele in paralel
  std::mt19937 mt = std::mt19937();
  mt.seed(time(NULL));

  //nonentropy_stresstest(mt);
  //entropy_convergencetest(mt);
  //dp_stresstest(mt);
  //small_sparse(mt);
  sparse_stresstest(mt);

  return 0;
}
