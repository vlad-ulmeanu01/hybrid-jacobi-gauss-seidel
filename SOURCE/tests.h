#ifndef __TESTS_H
#define __TESTS_H

#include "utilities.h"
#include "jacobi.h"
#include "gauss_seidel.h"
#include "sor.h"
#include "hybrid.h"
#include "entropy.h"
#include "entropy_dp.h"

void
one_simulation (std::mt19937 &mt, int n, int nt, double bucketsize_coefficient,
  double coef_a, double coef_b,
  bool should_generate_solution, bool should_calculate_radial_spectrum, bool should_solve_jacobi,
  bool should_solve_gauss_seidel, bool should_solve_jacobi_analytic_parallel,
  bool should_solve_gauss_seidel_analytic, bool should_hybrid_jacobi_gauss_seidel,
  bool should_solve_gauss_seidel_entropy, bool should_hybrid_entropy_jacobi_gauss_seidel,
  bool should_solve_sor_analytic, bool should_solve_gauss_seidel_entropy_dp,
  bool should_hybrid_entropy_dp,
  bool should_pass_solution, bool should_differentiate_generated_matrix_weight);

void
entropy_convergencetest (std::mt19937 &mt);

void
dp_stresstest (std::mt19937 &mt);

#endif
