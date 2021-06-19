#ifndef __ENTROPY_H
#define __ENTROPY_H

#include "utilities.h"
#include "hybrid.h"  /// pentru hybrid_parallel_function din functia hybrid_entropy_jacobi_gauss_seidel

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel_entropy (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol,
                            int max_pasi, bool should_pass_solution, Eigen::MatrixXd &x_precise);

std::pair<Eigen::MatrixXd, int>
hybrid_entropy_jacobi_gauss_seidel (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol,
                                    int max_pasi, double ct_bsz, bool should_pass_solution,
                                    Eigen::MatrixXd &x_precise);

#endif
