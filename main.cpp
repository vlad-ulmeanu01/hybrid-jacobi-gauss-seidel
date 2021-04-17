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
#include <Eigen/Dense>

//#include <omp.h>

/// g++ -std=c++17 -O3 -fopenmp -march=native -lpthread -I /usr/local/include/Eigen/ main.cpp -o main

///Explanation: Gauss-Seidel method is applicable to strictly diagonally dominant or
///symmetric positive definite matrices because only in this case convergence is possible.

std::pair<Eigen::MatrixXd, double>
generate_solvable_iterative_matrix(int n, bool type) {
  Eigen::MatrixXd W = Eigen::MatrixXd::Random(n, n).cwiseAbs();

  if (type == false) {
    Eigen::MatrixXd rand_coef = Eigen::MatrixXd::Random(1, n).cwiseAbs();

    ///matrice diagonal dominanta
    for (int i = 0; i < n; i++) {
      double oth = W.row(i).sum() - W(i, i);
      W(i, i) = (1 + 0.25 * rand_coef(0, i)) * oth;
    }
  } else {
    for (int i = 0; i < n; i++)
      for (int j = i+1; j < n; j++)
        W(i, j) = W(j, i);

    for (int i = 0; i < n; i++)
      W(i, i) *= n * 2;
  }


//  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(n, n);
//  for (int i = 0; i < n; i++)
//    N(i, i) = W(i, i);
//  Eigen::MatrixXd P = N - W, inv_N = N.inverse();
//  double rad_spectrum = (inv_N * P).eigenvalues().cwiseAbs().lpNorm<Eigen::Infinity>();

  double rad_spectrum = 0;

  return std::make_pair(W, rad_spectrum);
}

std::pair<Eigen::MatrixXd, int>
solve_jacobi (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi)
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
  }
  return std::make_pair(x, pasi);
}

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi)
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
  }
  return std::make_pair(x, pasi);
}

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel_analytic (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi)
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
      if (0 <= i-1)
        x(i, 0) -= A.row(i).segment(0, i-1) * x.col(0).segment(0, i-1);
      if (0 <= n-1-i)
        x(i, 0) -= A.row(i).segment(i+1, n-1-i) * x0.col(0).segment(i+1, n-1-i);
      x(i, 0) /= A(i, i);
    }
    pasi++;
  }

  return std::make_pair(x, pasi);
}

void
sort_matrix_lines(Eigen::MatrixXd &M, std::function<bool(Eigen::MatrixXd &, Eigen::MatrixXd &)> cmp)
{
  int n = M.rows();
  std::vector<Eigen::MatrixXd> linii;
  for (int i = 0; i < n; i++)
    linii.push_back(M.row(i));
  std::sort(linii.begin(), linii.end(), cmp);
  for (int i = 0; i < n; i++)
    M.row(i) = linii[i];
}

///schimba coloanele lui A intre ele ai ordinea lor sa fie cea din vectorul permutatie dat
///perm trebuie sa fie vector linie.
///trebuie sa schimb si ordinea lui b? nu e ca la inmultire de matrice.
///NU trebuie sa schimb ordinea lui A???????
void
shift_columns_to_match(Eigen::MatrixXd &b, Eigen::MatrixXd perm)
{
  int n = b.rows();
  std::vector<int> where(n);
  for (int i = 0; i < n; i++)
    where[b(i, 0)] = i;

  for (int i = 0; i < n; i++) {
    int tmp = where[perm(0, i)];
    if (tmp != i) {
      where[perm(0, i)] = i;
      where[b(i, 0)] = tmp;
      b.row(i).swap(b.row(tmp));
    }
  }
}

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel_entropy (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi)
{
  if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
    std::cerr << "solve_gauss_seidel_entropy invalid matrices\n";
    exit(0);
  }
  int n = A.rows();
  A.conservativeResize(n + 1, Eigen::NoChange);

  ///trebuie sa dau swap la coloanele lui A
  for (int i = 0; i < n; i++)
    A(n, i) = i;

  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, 3);
  for (int i = 0; i < n; i++)
    x(i, 2) = i;

  int pasi = 0;

  while (pasi == 0 || (pasi < max_pasi && (x.col(0) - x.col(1)).lpNorm<Eigen::Infinity>() > tol)) {
    x.col(1) = x.col(0);
    for (int i = 0; i < n; i++) {
      x(i, 0) = b(i, 0);
      if (0 <= i-1)
        x(i, 0) -= A.row(i).segment(0, i-1) * x.col(0).segment(0, i-1);
      if (0 <= n-1-i)
        x(i, 0) -= A.row(i).segment(i+1, n-1-i) * x.col(1).segment(i+1, n-1-i);
      x(i, 0) /= A(i, i);
    }
    pasi++;

    sort_matrix_lines(x, []
      (Eigen::MatrixXd &a_, Eigen::MatrixXd &b_) {
        return fabs(a_(0, 0) - a_(0, 1)) < fabs(b_(0, 0) - b_(0, 1));
      });
    shift_columns_to_match(b, x.col(2).transpose());
  }

  sort_matrix_lines(x, []
  (Eigen::MatrixXd &a_, Eigen::MatrixXd &b_) {
    return a_(0, 2) < b_(0, 2);
  });
  shift_columns_to_match(b, x.col(2).transpose());

  return std::make_pair(x.col(0), pasi);
}

void
hybrid_parallel_function (Eigen::MatrixXd &A, Eigen::MatrixXd &b, Eigen::MatrixXd &x,
                          Eigen::MatrixXd &x0, int buck_bound, int n, int start_itv, int end_itv)
{
  assert(start_itv >= 0 && start_itv < n && end_itv >= 0 && end_itv < n);
  for (int i = start_itv; i <= end_itv; i++) {
    x(i, 0) = b(i, 0);
    if (0 <= buck_bound) {
      x(i, 0) -= A.row(i).segment(0, buck_bound) * x.col(0).segment(0, buck_bound);
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

std::pair<Eigen::MatrixXd, int>
hybrid_jacobi_gauss_seidel (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi)
{
  const int num_workers = (int)std::thread::hardware_concurrency();

  int n = A.rows(), bucket_size = sqrt(n), remaining = 0;
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
  }
  return std::make_pair(x, pasi);
}

std::pair<Eigen::MatrixXd, int>
hybrid_entropy_jacobi_gauss_seidel (Eigen::MatrixXd A, Eigen::MatrixXd b, double tol, int max_pasi)
{
  const int num_workers = (int)std::thread::hardware_concurrency();

  int n = A.rows();
  A.conservativeResize(n + 1, Eigen::NoChange);

  ///trebuie sa dau swap la coloanele lui A
  for (int i = 0; i < n; i++)
    A(n, i) = i;

  int bucket_size = sqrt(n), remaining = 0;
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
      (Eigen::MatrixXd &a_, Eigen::MatrixXd &b_) {
        return fabs(a_(0, 0) - a_(0, 1)) < fabs(b_(0, 0) - b_(0, 1));
      });

    shift_columns_to_match(b, x.col(2).transpose());
  }

  ///aduc liniile inapoi la ordinea originala
  sort_matrix_lines(x, []
  (Eigen::MatrixXd &a_, Eigen::MatrixXd &b_) {
    return a_(0, 2) < b_(0, 2);
  });
  shift_columns_to_match(b, x.col(2).transpose());

  return std::make_pair(x.col(0), pasi);
}

int
main()
{
  Eigen::initParallel(); ///nu ajuta, dar ar tb pus pt ca folosesc si eu op mele in paralel

  auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();

  srand(time(NULL));

  int n;
  std::cin >> n;

  auto A_r = generate_solvable_iterative_matrix(n, false);
  std::cout << "Matricea A are Rho: " << A_r.second << '\n';

  Eigen::MatrixXd A = A_r.first;
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(n, 1).cwiseAbs();


  start = std::chrono::steady_clock::now();
  Eigen::MatrixXd x_precise = A.colPivHouseholderQr().solve(b);
  stop = std::chrono::steady_clock::now();
  auto duration_hholder = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "timp hholder: " << duration_hholder.count() / 1000000.0 << "s\n-------------\n";
//  Eigen::MatrixXd x_precise = Eigen::MatrixXd::Zero(n, 1);


  start = std::chrono::steady_clock::now();
  auto jacobi = solve_jacobi(A, b, 0.00001, 10000);

  stop = std::chrono::steady_clock::now();
  auto duration_jacobi = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  Eigen::MatrixXd x_jacobi = jacobi.first;
  int pasi_jacobi = jacobi.second;

  std::cout << (x_jacobi - x_precise).lpNorm<Eigen::Infinity>() << '\n';
  std::cout << "pasi jacobi: " << pasi_jacobi << '\n';
  std::cout << "timp jacobi: " << duration_jacobi.count() / 1000000.0 << "s\n-------------\n";



  start = std::chrono::steady_clock::now();
  auto gauss_seidel = solve_gauss_seidel(A, b, 0.00001, 10000);

  stop = std::chrono::steady_clock::now();
  auto duration_gauss_seidel = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  Eigen::MatrixXd x_gauss_seidel = gauss_seidel.first;
  int pasi_gauss_seidel = gauss_seidel.second;

  std::cout << (x_gauss_seidel - x_precise).lpNorm<Eigen::Infinity>() << '\n';
  std::cout << "pasi gauss_seidel: " << pasi_gauss_seidel << '\n';
  std::cout << "timp gauss_seidel: " << duration_gauss_seidel.count() / 1000000.0 << "s\n-------------\n";



  start = std::chrono::steady_clock::now();
  auto gauss_seidel_analytic = solve_gauss_seidel_analytic(A, b, 0.00001, 10000);

  stop = std::chrono::steady_clock::now();
  auto duration_gauss_seidel_analytic = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  Eigen::MatrixXd x_gauss_seidel_analytic = gauss_seidel_analytic.first;
  int pasi_gauss_seidel_analytic = gauss_seidel_analytic.second;

  std::cout << (x_gauss_seidel_analytic - x_precise).lpNorm<Eigen::Infinity>() << '\n';
  std::cout << "pasi gauss_seidel_analytic: " << pasi_gauss_seidel_analytic << '\n';
  std::cout << "timp gauss_seidel_analytic: " << duration_gauss_seidel_analytic.count() / 1000000.0 << "s\n-------------\n";



  start = std::chrono::steady_clock::now();
  auto hybrid = hybrid_jacobi_gauss_seidel(A, b, 0.00001, 10000);

  stop = std::chrono::steady_clock::now();
  auto duration_hybrid = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  Eigen::MatrixXd x_hybrid = hybrid.first;
  int pasi_hybrid = hybrid.second;

  std::cout << (x_hybrid - x_precise).lpNorm<Eigen::Infinity>() << '\n';
  std::cout << "pasi hibrid: " << pasi_hybrid << '\n';
  std::cout << "timp hibrid: " << duration_hybrid.count() / 1000000.0 << "s\n-------------\n";



  start = std::chrono::steady_clock::now();
  auto gs_entropy = solve_gauss_seidel_entropy(A, b, 0.00001, 5);

  stop = std::chrono::steady_clock::now();
  auto duration_gs_entropy = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  Eigen::MatrixXd x_gs_entropy = gs_entropy.first;
  int pasi_gs_entropy = gs_entropy.second;

  std::cout << (x_gs_entropy - x_precise).lpNorm<Eigen::Infinity>() << '\n';
  std::cout << "pasi gs entropy: " << pasi_gs_entropy << '\n';
  std::cout << "timp gs entropy: " << duration_gs_entropy.count() / 1000000.0 << "s\n-------------\n";


  start = std::chrono::steady_clock::now();
  auto hybrid_entropy = hybrid_entropy_jacobi_gauss_seidel(A, b, 0.00001, 5);

  stop = std::chrono::steady_clock::now();
  auto duration_hybrid_entropy = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  Eigen::MatrixXd x_hybrid_entropy = hybrid_entropy.first;
  int pasi_hybrid_entropy = hybrid_entropy.second;

  std::cout << (x_hybrid_entropy - x_precise).lpNorm<Eigen::Infinity>() << '\n';
  std::cout << "pasi hibrid entropy: " << pasi_hybrid_entropy << '\n';
  std::cout << "timp hibrid entropy: " << duration_hybrid_entropy.count() / 1000000.0 << "s\n-------------\n";

  return 0;
}
