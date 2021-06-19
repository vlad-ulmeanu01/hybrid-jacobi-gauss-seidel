#include "entropy.h"

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel_entropy (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol,
                            int max_pasi, bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
    std::cerr << "solve_gauss_seidel_entropy invalid matrices\n";
    exit(0);
  }

  int n = A.cols();
  A.conservativeResize(n+1, Eigen::NoChange);
  for (int i = 0; i < n; i++)
    A(n, i) = i;

  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, 3);
  for (int i = 0; i < n; i++)
    x(i, 2) = i;

  int pasi = 0;

  while (pasi == 0 || (pasi < max_pasi && (x.col(0) - x.col(1)).lpNorm<Eigen::Infinity>() > tol)) {
    if (DEBUG) {
      std::cout << "DBG GSe pas nr " << pasi << " Rho(A): " << calculate_radial_spectrum(mt, A) << '\n';
    }

    x.col(1) = x.col(0);
    for (int i = 0; i < n; i++) {
      x(i, 0) = b(i, 0);
      if (0 <= i) ///!! nu i-1...
        x(i, 0) -= A.row(i).segment(0, i) * x.col(0).segment(0, i);
      if (0 <= n-1-i)
        x(i, 0) -= A.row(i).segment(i+1, n-1-i) * x.col(1).segment(i+1, n-1-i);
      x(i, 0) /= A(i, i);
    }
    pasi++;

    sort_matrix_lines(x, []
      (const Eigen::MatrixXd &a_, const Eigen::MatrixXd &b_) {
        return fabs(a_(0, 0) - a_(0, 1)) < fabs(b_(0, 0) - b_(0, 1));
      });
    shift_columns_to_match(A, b, x.col(2).transpose(), should_pass_solution, x_precise);

    print_error_to_solution(pasi, should_pass_solution, "gauss_seidel_entropy", x.col(0), x.col(1), x_precise);
  }

  sort_matrix_lines(x, []
  (const Eigen::MatrixXd &a_, const Eigen::MatrixXd &b_) {
    return a_(0, 2) < b_(0, 2);
  });
  shift_columns_to_match(A, b, x.col(2).transpose(), should_pass_solution, x_precise);

  return std::make_pair(x.col(0), pasi);
}

///ct_bsz * sqrt(n) == bucket_size. bucket_size < 1 ar fi ideal, dar exista overhead pt calcul in paralel
std::pair<Eigen::MatrixXd, int>
hybrid_entropy_jacobi_gauss_seidel (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol,
                                    int max_pasi, double ct_bsz, bool should_pass_solution,
                                    Eigen::MatrixXd &x_precise)
{
  if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
    std::cerr << "hybrid_entropy_jacobi_gauss_seidel invalid matrices\n";
    exit(0);
  }

  const int num_workers = (int)std::thread::hardware_concurrency();

  int n = A.cols();
  A.conservativeResize(n+1, Eigen::NoChange);
  for (int i = 0; i < n; i++)
    A(n, i) = i;

  int bucket_size = ct_bsz * sqrt(n), remaining = 0;
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

  ///coloana 2 este x0, coloana 3 contine indicii liniilor
  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, 3);
  for (int i = 0; i < n; i++)
    x(i, 2) = i;
  Eigen::MatrixXd x_c, x0_c;

  std::vector<std::thread> threads;
  int pasi = 0, buck_bound, worker_length, start_itv, end_itv;

  while (pasi == 0 || (pasi < max_pasi && (x.col(0) - x.col(1)).lpNorm<Eigen::Infinity>() > tol)) {
    if (DEBUG) {
      std::cout << "DBG HGSe pas nr " << pasi << " Rho(A): " << calculate_radial_spectrum(mt, A) << '\n';
    }

    x.col(1) = x.col(0);
    x_c = x.col(0);
    x0_c = x.col(1);

    for (int current_bucket = 0; current_bucket < (int)ends_of_buckets.size(); current_bucket++) {
      buck_bound = -1;
      if (current_bucket > 0)
        buck_bound = ends_of_buckets[current_bucket - 1];

      worker_length = (ends_of_buckets[current_bucket] - buck_bound + num_workers - 1) / num_workers;

      threads.clear();
      for (int i = 0; i < num_workers; i++) {
        start_itv = std::min(buck_bound + 1 + i * worker_length, ends_of_buckets[current_bucket]);
        end_itv = std::min(buck_bound + 1 + (i+1) * worker_length - 1, ends_of_buckets[current_bucket]);

        std::thread th(hybrid_parallel_function, std::ref(A), std::ref(b),
                       std::ref(x_c), std::ref(x0_c),
                       buck_bound, n, start_itv, end_itv);
        threads.push_back(std::move(th));
      }

      for (int i = 0; i < num_workers; i++)
        threads[i].join();
    }
    pasi++;

    x.col(0) = x_c;
    x.col(1) = x0_c;

    ///sortez liniile ai cele care s-au modificat cel mai mult sa fie la coada (sa profite mai mult la
    ///urmatoarele iteratii)
    sort_matrix_lines(x, []
      (const Eigen::MatrixXd &a_, const Eigen::MatrixXd &b_) {
        return fabs(a_(0, 0) - a_(0, 1)) < fabs(b_(0, 0) - b_(0, 1));
      });

    shift_columns_to_match(A, b, x.col(2).transpose(), should_pass_solution, x_precise);

    print_error_to_solution(pasi, should_pass_solution, "hibrid_entropy", x.col(0), x.col(1), x_precise);
  }

  ///aduc liniile inapoi la ordinea originala
  sort_matrix_lines(x, []
  (const Eigen::MatrixXd &a_, const Eigen::MatrixXd &b_) {
    return a_(0, 2) < b_(0, 2);
  });
  shift_columns_to_match(A, b, x.col(2).transpose(), should_pass_solution, x_precise);

  return std::make_pair(x.col(0), pasi);
}
