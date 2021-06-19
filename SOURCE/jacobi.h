#ifndef __JACOBI_H
#define __JACOBI_H

#include "utilities.h"

std::pair<Eigen::MatrixXd, int>
solve_jacobi (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
              bool should_pass_solution, Eigen::MatrixXd &x_precise);

void
jacobi_parallel_function (Eigen::MatrixXd &A, Eigen::MatrixXd &b, Eigen::MatrixXd &x,
                          Eigen::MatrixXd &x0, int n, int start_itv, int end_itv);

std::pair<Eigen::MatrixXd, int>
solve_jacobi_analytic_parallel (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
                                bool should_pass_solution, Eigen::MatrixXd &x_precise);

#endif
