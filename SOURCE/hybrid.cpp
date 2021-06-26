#include "hybrid.h"

void
hybrid_parallel_function (Eigen::MatrixXd &A, Eigen::MatrixXd &b, Eigen::MatrixXd &x,
                          Eigen::MatrixXd &x0, int buck_bound, int n, int start_itv, int end_itv)
{
  if (start_itv > end_itv)
    return;

  assert(start_itv >= 0 && start_itv < n && end_itv >= 0 && end_itv < n);
  for (int i = start_itv; i <= end_itv; i++) {
    x(i, 0) = b(i, 0);
    if (0 <= buck_bound) {
      x(i, 0) -= A.row(i).segment(0, buck_bound+1) * x.col(0).segment(0, buck_bound+1); ///??buck_bound + 1 inl de buck_bound?
    }
    if (buck_bound+1 <= i-1) {
      x(i, 0) -= A.row(i).segment(buck_bound+1, i-1 - buck_bound) *
                 x0.col(0).segment(buck_bound+1, i-1 - buck_bound);
    }
    if (i+1 <= n-1) {
      x(i, 0) -= A.row(i).segment(i+1, n-1-i) * x0.col(0).segment(i+1, n-1-i);
    }
    x(i, 0) /= A(i, i);
  }
}

///ct_bsz * sqrt(n) == bucket_size. bucket_size < 1 ar fi ideal, dar exista overhead pt calcul in paralel
/// cea mai rapida functie de aici, posibil cea mai buna
std::pair<Eigen::MatrixXd, int>
hybrid_jacobi_gauss_seidel (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi, double ct_bsz,
                            bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  const int num_workers = (int)std::thread::hardware_concurrency();

  int n = A.rows(), bucket_size = ct_bsz * sqrt(n), remaining = 0;
  std::vector<int> ends_of_buckets;

  for (int i = 0; i < n; i++) {
    if (i == 0) {
      remaining = bucket_size;
    } else if (remaining == 0) {
      remaining = bucket_size;
      ends_of_buckets.push_back(i-1);
    }
    remaining--;
  }
  ends_of_buckets.push_back(n-1);

  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, 1), x0 = x;
  std::vector<std::thread> threads;
  int pasi = 0, buck_bound, worker_length, start_itv, end_itv;

  while (pasi == 0 || (pasi < max_pasi && (x - x0).lpNorm<Eigen::Infinity>() > tol)) {
    x0 = x;
    for (int current_bucket = 0; current_bucket < (int)ends_of_buckets.size(); current_bucket++) {
      buck_bound = -1;
      if (current_bucket > 0)
        buck_bound = ends_of_buckets[current_bucket - 1];

      worker_length = (ends_of_buckets[current_bucket] - buck_bound + num_workers - 1) / num_workers;

      threads.clear();
      for (int i = 0; i < num_workers; i++) {
        start_itv = std::min(buck_bound + 1 + i * worker_length, ends_of_buckets[current_bucket]);
        end_itv = std::min(buck_bound + 1 + (i+1) * worker_length - 1, ends_of_buckets[current_bucket]);

        std::thread th(hybrid_parallel_function, std::ref(A), std::ref(b), std::ref(x), std::ref(x0),
                       buck_bound, n, start_itv, end_itv);
        threads.push_back(std::move(th));
      }

      for (int i = 0; i < num_workers; i++)
        threads[i].join();
    }
    pasi++;
    print_error_to_solution(pasi, should_pass_solution, "hibrid", x, x0, x_precise);
  }
  return std::make_pair(x, pasi);
}
