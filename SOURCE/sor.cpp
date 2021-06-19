#include "sor.h"

std::pair<Eigen::MatrixXd, int>
solve_sor_analytic (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
                    std::string str_print_err,
                    bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
    std::cerr << "solve_sor_analytic invalid matrices\n";
    exit(0);
  }
  int n = A.rows();
  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++)
    N(i, i) = A(i, i);

  Eigen::MatrixXd G_jacobi = N.inverse() * (N - A);
//  double Rho_jacobi = spectral_radius(mt, G_jacobi);
//  double w_part = Rho_jacobi / (1 + sqrt(1 - Rho_jacobi * Rho_jacobi));
//  double w = 1 + w_part * w_part;

  std::cout << "G_jacobi has " << G_jacobi.eigenvalues().imag().lpNorm<Eigen::Infinity>() << " biggest C-R part\n";

  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n), U = Eigen::MatrixXd::Zero(n, n), D = N;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i < j) {
        U(i, j) = A(i, j);
      } else if (i > j){
        L(i, j) = A(i, j);
      }

  double w = -1.0, best_Rho = 1.0 * (1<<30);
  std::uniform_real_distribution<double> distr(0.7, 0.85);
  for (int z = 0; z < 25; z++) {
    double now_w = distr(mt);
    double Rho = spectral_radius(mt, (D + now_w * L).inverse() * ((1 - now_w) * D - now_w * U));
    if (z == 0 || best_Rho > Rho) {
      best_Rho = Rho;
      w = now_w;
    }
  }

  str_print_err += "do your thing mingw"; //std::to_string(w);

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

      x(i, 0) = x(i, 0) * w + x0(i, 0) * (1 - w);
    }
    pasi++;
    print_error_to_solution(pasi, should_pass_solution, str_print_err, x, x0, x_precise);
  }

  return std::make_pair(x, pasi);
}
