#ifndef __GAUSS_SEIDEL_H
#define __GAUSS_SEIDEL_H

#include "utilities.h"

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
                    bool should_pass_solution, Eigen::MatrixXd &x_precise);

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel_analytic (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
                             bool should_pass_solution, Eigen::MatrixXd &x_precise);

#endif
