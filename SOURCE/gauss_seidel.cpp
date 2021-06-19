#include "gauss_seidel.h"

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
                    bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
    std::cerr << "solve_gauss_seidel invalid matrices\n";
    exit(0);
  }
  int n = A.rows();
  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i >= j) N(i, j) = A(i, j);

  Eigen::MatrixXd P = N - A, inv_N = N.inverse();
  Eigen::MatrixXd G = inv_N * P, c = inv_N * b;

  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, 1), x0 = x;
  int pasi = 0;
  while (pasi == 0 || (pasi < max_pasi && (x - x0).lpNorm<Eigen::Infinity>() > tol)) {
    x0 = x;
    x = G * x0 + c;
    pasi++;
    print_error_to_solution(pasi, should_pass_solution, "gauss_seidel_matriceal", x, x0, x_precise);
  }
  return std::make_pair(x, pasi);
}

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel_analytic (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
                             bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
    std::cerr << "solve_gauss_seidel_analytic invalid matrices\n";
    exit(0);
  }
  int n = A.rows();

  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, 1), x0 = x;
  int pasi = 0;
  while (pasi == 0 || (pasi < max_pasi && (x - x0).lpNorm<Eigen::Infinity>() > tol)) {
    x0 = x;
    for (int i = 0; i < n; i++) {
      x(i, 0) = b(i, 0);
      if (0 <= i) ///!!nu i-1..
        x(i, 0) -= A.row(i).segment(0, i) * x.col(0).segment(0, i);
      if (0 <= n-1-i)
        x(i, 0) -= A.row(i).segment(i+1, n-1-i) * x0.col(0).segment(i+1, n-1-i);
      x(i, 0) /= A(i, i);
    }
    pasi++;
    print_error_to_solution(pasi, should_pass_solution, "gauss_seidel_analytic", x, x0, x_precise);
  }

  return std::make_pair(x, pasi);
}
