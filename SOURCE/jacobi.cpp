#include "jacobi.h"

std::pair<Eigen::MatrixXd, int>
solve_jacobi (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
              bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
    std::cerr << "solve_jacobi invalid matrices\n";
    exit(0);
  }
  int n = A.rows();
  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++)
    N(i, i) = A(i, i);

  Eigen::MatrixXd P = N - A, inv_N = N.inverse();
  Eigen::MatrixXd G = inv_N * P, c = inv_N * b;

  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, 1), x0 = x;
  int pasi = 0;
  while (pasi == 0 || (pasi < max_pasi && (x - x0).lpNorm<Eigen::Infinity>() > tol)) {
    x0 = x;
    x = G * x0 + c;
    pasi++;
    print_error_to_solution(pasi, should_pass_solution, "jacobi_matriceal", x, x0, x_precise);
  }
  return std::make_pair(x, pasi);
}

void
jacobi_parallel_function (Eigen::MatrixXd &A, Eigen::MatrixXd &b, Eigen::MatrixXd &x,
                          Eigen::MatrixXd &x0, int n, int start_itv, int end_itv)
{
  if (start_itv > end_itv)
    return;

  assert(start_itv >= 0 && start_itv < n && end_itv >= 0 && end_itv < n);
  for (int i = start_itv; i <= end_itv; i++) {
    x(i, 0) = b(i, 0);
    x(i, 0) -= A.row(i).segment(0, n) * x0.col(0).segment(0, n);
    x(i, 0) += A(i, i) * x0(i, 0);
    x(i, 0) /= A(i, i);
  }
}

std::pair<Eigen::MatrixXd, int>
solve_jacobi_analytic_parallel (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
                                bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  const int num_workers = (int)std::thread::hardware_concurrency();
  int n = A.rows();

  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, 1), x0 = x;
  std::vector<std::thread> threads;
  int pasi = 0, worker_length, start_itv, end_itv;

  worker_length = (n + num_workers - 1) / num_workers;

  while (pasi == 0 || (pasi < max_pasi && (x - x0).lpNorm<Eigen::Infinity>() > tol)) {
    x0 = x;

    threads.clear();
    for (int i = 0; i < num_workers; i++) {
      start_itv = i * worker_length;
      end_itv = std::min((i+1) * worker_length - 1, n-1);

      std::thread th(jacobi_parallel_function, std::ref(A), std::ref(b), std::ref(x), std::ref(x0),
                     n, start_itv, end_itv);
      threads.push_back(std::move(th));
    }

    for (int i = 0; i < num_workers; i++)
      threads[i].join();

    pasi++;
    print_error_to_solution(pasi, should_pass_solution, "jacobi_parallel", x, x0, x_precise);
  }
  return std::make_pair(x, pasi);
}
