#ifndef __SOR_H
#define __SOR_H

#include "utilities.h"

std::pair<Eigen::MatrixXd, int>
solve_sor_analytic (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
                    std::string str_print_err,
                    bool should_pass_solution, Eigen::MatrixXd &x_precise);

#endif
