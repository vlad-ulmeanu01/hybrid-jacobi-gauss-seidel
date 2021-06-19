#ifndef __ENTROPY_DP_H
#define __ENTROPY_DP_H

#include "utilities.h"
#include "hybrid.h"

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel_entropy_dp (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol,
                               int max_pasi, bool should_pass_solution, Eigen::MatrixXd &x_precise);

std::pair<Eigen::MatrixXd, int>
hybrid_entropy_dp (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
                   double ct_bsz, bool should_pass_solution, Eigen::MatrixXd &x_precise);

#endif
