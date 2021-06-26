#include "entropy_dp.h"

/// cea mai precisa functie de aici. posibil cea mai buna.
std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel_entropy_dp (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol,
                               int max_pasi, bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
    std::cerr << "solve_gauss_seidel_entropy_dp invalid matrices\n";
    exit(0);
  }

  ////TIMER
  /**/auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();
  /**/auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  /**/double time_spent_sorting_matrix_lines = 0, time_spent_sorting_shift_columns = 0;
  /**/double time_spent_transitioning = 0, time_spent_other = 0;
  ////TIMER

  int n = A.cols();
  A.conservativeResize(n+1, Eigen::NoChange);
  for (int i = 0; i < n; i++)
    A(n, i) = i;

  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, 3);
  for (int i = 0; i < n; i++)
    x(i, 2) = i;

  bool did_sort_on_last_step = false;

  ///d[i] = cea mai buna stare dupa i pasi.
  ///o stare contine A, b, x_precise si x asociate. o stare e mai buna decat alta daca "distanta"
  ///dintre x.col(0) si x.col(1) e mai mica.

  std::function<void(Eigen::MatrixXd &, Eigen::MatrixXd &, Eigen::MatrixXd &)> perform_transition = [&]
  (Eigen::MatrixXd &x_, Eigen::MatrixXd &A_, Eigen::MatrixXd &b_) {
    for (int i = 0; i < n; i++) {
      x_(i, 0) = b_(i, 0);
      if (0 <= i)
        x_(i, 0) -= A_.row(i).segment(0, i) * x_.col(0).segment(0, i);
      if (0 <= n-1-i)
        x_(i, 0) -= A_.row(i).segment(i+1, n-1-i) * x_.col(1).segment(i+1, n-1-i);
      x_(i, 0) /= A_(i, i);
    }
  };

  int pasi;
  for (pasi = 1; pasi <= max_pasi; pasi++) {
    if (DEBUG) { ////TIMER
      /**/start = std::chrono::steady_clock::now();
    } ////TIMER

    Eigen::MatrixXd A_cpy = A, b_cpy = b, x_precise_cpy = x_precise, x_cpy = x;

    if (DEBUG) { ////TIMER
      /**/stop = std::chrono::steady_clock::now();
      /**/duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      /**/time_spent_other += duration.count() / 1000000.0;
      /**/start = std::chrono::steady_clock::now();
    } ////TIMER

    ///fac trecere cu sortare pe cpy.
    sort_matrix_lines(x_cpy, []
        (const Eigen::MatrixXd &a_, const Eigen::MatrixXd &b_) {
          return fabs(a_(0, 0) - a_(0, 1)) < fabs(b_(0, 0) - b_(0, 1));
        });

    if (DEBUG) { ////TIMER
      /**/stop = std::chrono::steady_clock::now();
      /**/duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      /**/time_spent_sorting_matrix_lines += duration.count() / 1000000.0;
      /**/start = std::chrono::steady_clock::now();
    } ////TIMER

    /// aduc si x_precise_cpy la forma dorita.
    shift_columns_to_match(A_cpy, b_cpy, x_cpy.col(2).transpose(), should_pass_solution, x_precise_cpy);

    if (DEBUG) { ////TIMER
      /**/stop = std::chrono::steady_clock::now();
      /**/duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      /**/time_spent_sorting_shift_columns += duration.count() / 1000000.0;
      /**/start = std::chrono::steady_clock::now();
    } ////TIMER

    ///acum fac trecere directa si pe normal si pe copie.
    x.col(1) = x.col(0);
    x_cpy.col(1) = x_cpy.col(0);

    std::thread th1(perform_transition, std::ref(x), std::ref(A), std::ref(b));
    std::thread th2(perform_transition, std::ref(x_cpy), std::ref(A_cpy), std::ref(b_cpy));

    if (DEBUG) { ////TIMER
      /**/stop = std::chrono::steady_clock::now();
      /**/duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      /**/time_spent_transitioning += duration.count() / 1000000.0;
      /**/start = std::chrono::steady_clock::now();
    } ////TIMER

    th1.join();
    th2.join();

//    if ((x.col(0) - x_precise).lpNorm<Eigen::Infinity>() >
//        (x_cpy.col(0) - x_precise_cpy).lpNorm<Eigen::Infinity>()) {
    if (pasi <= max_pasi &&
        (x.col(0) - x.col(1)).lpNorm<Eigen::Infinity>() <
        (x_cpy.col(0) - x_cpy.col(1)).lpNorm<Eigen::Infinity>()) { ///norma 2 e mult mai slaba??
      did_sort_on_last_step = true;
      A = A_cpy;
      b = b_cpy;
      x_precise = x_precise_cpy;
      x = x_cpy;
    } else {
      did_sort_on_last_step = false;
    }

    if (did_sort_on_last_step == false)
      print_error_to_solution(pasi, should_pass_solution, "gauss_seidel_entropy_dp_0",
                              x.col(0), x.col(1), x_precise);
    else
      print_error_to_solution(pasi, should_pass_solution, "gauss_seidel_entropy_dp_1",
                              x.col(0), x.col(1), x_precise);

    if (DEBUG) { ////TIMER
      /**/stop = std::chrono::steady_clock::now();
      /**/duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      /**/time_spent_other += duration.count() / 1000000.0;
    } ////TIMER

    if ((x.col(0) - x.col(1)).lpNorm<Eigen::Infinity>() <= tol)
      break;
  }

  if (DEBUG) { ////TIMER
    /**/start = std::chrono::steady_clock::now();
  } ////TIMER

  sort_matrix_lines(x, []
  (const Eigen::MatrixXd &a_, const Eigen::MatrixXd &b_) {
    return a_(0, 2) < b_(0, 2);
  });

  if (DEBUG) { ////TIMER
    /**/stop = std::chrono::steady_clock::now();
    /**/duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    /**/time_spent_sorting_matrix_lines += duration.count() / 1000000.0;
    /**/start = std::chrono::steady_clock::now();
  } ////TIMER

  shift_columns_to_match(A, b, x.col(2).transpose(), should_pass_solution, x_precise);

  if (DEBUG) { ////TIMER
    /**/stop = std::chrono::steady_clock::now();
    /**/duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    /**/time_spent_sorting_shift_columns += duration.count() / 1000000.0;

    std::cout << "gs_edp: time spent sorting matrix lines : " << time_spent_sorting_matrix_lines << '\n';
    std::cout << "gs_edp: time spent sorting shift columns: " << time_spent_sorting_shift_columns << '\n';
    std::cout << "gs_edp: time spent transitioning        : " << time_spent_transitioning << '\n';
    std::cout << "gs_edp: time spent other                : " << time_spent_other << '\n';
  } ////TIMER

  return std::make_pair(x.col(0), pasi);
}

/// mai rapida decat solve_gauss_seidel_entropy_dp, dar mai putin precisa. oricum ambele au constanta proasta a lui n^2.
std::pair<Eigen::MatrixXd, int>
hybrid_entropy_dp (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi,
                   double ct_bsz, bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
    std::cerr << "hybrid_entropy_dp invalid matrices\n";
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

  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, 3);
  for (int i = 0; i < n; i++)
    x(i, 2) = i;

  bool did_sort_on_last_step = false;
  std::vector<std::thread> threads;
  int pasi, buck_bound, worker_length, start_itv, end_itv;

  auto perform_parallel_on_x = [&] (Eigen::MatrixXd &A_, Eigen::MatrixXd &b_, Eigen::MatrixXd &x_) {
    Eigen::MatrixXd x_col = x_.col(0), x0_col = x_.col(1);

    for (int current_bucket = 0; current_bucket < (int)ends_of_buckets.size(); current_bucket++) {
      buck_bound = -1;
      if (current_bucket > 0)
        buck_bound = ends_of_buckets[current_bucket - 1];

      worker_length = (ends_of_buckets[current_bucket] - buck_bound + num_workers - 1) / num_workers;

      threads.clear();
      for (int i = 0; i < num_workers; i++) {
        start_itv = std::min(buck_bound + 1 + i * worker_length, ends_of_buckets[current_bucket]);
        end_itv = std::min(buck_bound + 1 + (i+1) * worker_length - 1, ends_of_buckets[current_bucket]);

        std::thread th(hybrid_parallel_function, std::ref(A_), std::ref(b_),
                       std::ref(x_col), std::ref(x0_col),
                       buck_bound, n, start_itv, end_itv);
        threads.push_back(std::move(th));
      }

      for (int i = 0; i < num_workers; i++)
        threads[i].join();
    }
    x_.col(0) = x_col;
    x_.col(1) = x0_col;
  };

  for (pasi = 1; pasi <= max_pasi; pasi++) {
    Eigen::MatrixXd A_cpy = A, b_cpy = b, x_precise_cpy = x_precise, x_cpy = x;

    ///fac trecere cu sortare pe cpy.
    sort_matrix_lines(x_cpy, []
        (const Eigen::MatrixXd &a_, const Eigen::MatrixXd &b_) {
          return fabs(a_(0, 0) - a_(0, 1)) < fabs(b_(0, 0) - b_(0, 1));
        });

    /// aduc si x_precise_cpy la forma dorita.
    shift_columns_to_match(A_cpy, b_cpy, x_cpy.col(2).transpose(), should_pass_solution, x_precise_cpy);

    ///acum fac trecere hibrida si pe normal si pe copie.
    x.col(1) = x.col(0);
    x_cpy.col(1) = x_cpy.col(0); ///!! liniile astea doua.

    perform_parallel_on_x(A, b, x);
    perform_parallel_on_x(A_cpy, b_cpy, x_cpy);

    if ((x.col(0) - x.col(1)).lpNorm<Eigen::Infinity>() <
        (x_cpy.col(0) - x_cpy.col(1)).lpNorm<Eigen::Infinity>()) {
      did_sort_on_last_step = true;
      A = A_cpy;
      b = b_cpy;
      x_precise = x_precise_cpy;
      x = x_cpy;
    } else {
      did_sort_on_last_step = false;
    }

    if (did_sort_on_last_step == false)
      print_error_to_solution(pasi, should_pass_solution, "hybrid_entropy_dp_0",
                              x.col(0), x.col(1), x_precise);
    else
      print_error_to_solution(pasi, should_pass_solution, "hybrid_entropy_dp_1",
                              x.col(0), x.col(1), x_precise);

    if ((x.col(0) - x.col(1)).lpNorm<Eigen::Infinity>() <= tol)
      break;
  }

  sort_matrix_lines(x, []
  (const Eigen::MatrixXd &a_, const Eigen::MatrixXd &b_) {
    return a_(0, 2) < b_(0, 2);
  });
  shift_columns_to_match(A, b, x.col(2).transpose(), should_pass_solution, x_precise);

  return std::make_pair(x.col(0), pasi);
}
