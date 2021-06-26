#ifndef __UTILITIES_H
#define __UTILITIES_H

//#define EIGEN_DONT_PARALLELIZE
#include <iostream>
#include <utility>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cassert>
#include <thread>
#include <complex>
#include <Eigen/Dense>
//#include <omp.h>

#define DEBUG 0

void
print_error_to_solution (int pasi, bool should_pass_solution, std::string text,
                         const Eigen::MatrixXd &x, const Eigen::MatrixXd &x0, Eigen::MatrixXd &x_precise);

Eigen::MatrixXd
generate_random_matrix_mt (std::mt19937 &mt, int n, int m, double mi, double Ma);

void
populate_sparse_matrix
(std::mt19937 &mt, Eigen::MatrixXd &A, Eigen::MatrixXd &x_precise, Eigen::MatrixXd &b, double p, double mi, double Ma);

double
spectral_radius (std::mt19937 &mt, Eigen::MatrixXd A_e);

Eigen::MatrixXd
inverse_inferior_triangular (Eigen::MatrixXd A);

double
calculate_radial_spectrum (std::mt19937 &mt, Eigen::MatrixXd &W);

std::pair<Eigen::MatrixXd, double>
generate_solvable_iterative_matrix
(std::mt19937 &mt, int n, bool type, double coef_a, bool should_calculate_radial_spectrum);

void
sort_matrix_lines(Eigen::MatrixXd &M,
                  std::function<bool(const Eigen::MatrixXd &, const Eigen::MatrixXd &)> cmp);

void
shift_columns_to_match(Eigen::MatrixXd &A, Eigen::MatrixXd &b, const Eigen::MatrixXd &perm,
                       bool should_pass_solution, Eigen::MatrixXd &x_precise);

#endif
