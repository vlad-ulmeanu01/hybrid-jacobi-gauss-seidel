#ifndef __HYBRID_H
#define __HYBRID_H

#include "utilities.h"

void
hybrid_parallel_function (Eigen::MatrixXd &A, Eigen::MatrixXd &b, Eigen::MatrixXd &x,
                          Eigen::MatrixXd &x0, int buck_bound, int n, int start_itv, int end_itv);

std::pair<Eigen::MatrixXd, int>
hybrid_jacobi_gauss_seidel (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi, double ct_bsz,
                            bool should_pass_solution, Eigen::MatrixXd &x_precise);

#endif
