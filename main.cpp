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

/// g++ -std=c++11 -O3 -fopenmp -march=native -lpthread -I ~/Eigen/ main.cpp -o main
/// g++ -std=c++17 -O3 -fopenmp -march=native -lpthread -I /usr/local/include/Eigen/ main.cpp -o main

///Explanation: Gauss-Seidel method is applicable to strictly diagonally dominant or
///symmetric positive definite matrices because only in this case convergence is possible.

namespace Eigen {
  void operator ^= (Eigen::MatrixXd &A, int e) {
    assert(A.rows() == A.cols() && e >= 0);
    int n = A.rows();
    Eigen::MatrixXd Ans = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; i++)
      Ans(i, i) = 1;
    while (e) {
      if (e&1)
        Ans *= A;
      e >>= 1;
      A *= A;
    }
    A = Ans;
  }
}

void
print_error_to_solution (int pasi, bool should_pass_solution, std::string text,
                         const Eigen::MatrixXd &x, const Eigen::MatrixXd &x0, Eigen::MatrixXd &x_precise)
{
  if (!should_pass_solution)
    return;
  std::cout << pasi << " pasi; err_fata_de_solutie " << text << ": ";
  std::cout << (x - x_precise).lpNorm<Eigen::Infinity>() << '\n';
  std::cout << pasi << " pasi; err_fata_de_ult_val " << text << ": ";
  std::cout << (x - x0).lpNorm<Eigen::Infinity>() << '\n';
  std::cout << std::flush;
}

Eigen::MatrixXd generate_random_matrix_mt (std::mt19937 &mt, int n, int m, double mi, double Ma) {
  std::uniform_real_distribution<double> distr(mi, Ma);
  Eigen::MatrixXd A(n, m);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      A(i, j) = distr(mt);
  return A;
}

double
spectral_radius (std::mt19937 &mt, Eigen::MatrixXd A_e)
{
  assert(A_e.rows() == A_e.cols());
  int e = 100, n = A_e.rows();
  Eigen::MatrixXd A_e1 = A_e;
  A_e ^= e; ///A^e
  A_e1 *= A_e; ///A^(e+1) => lb_1 = (A_e1 * u)(1) / (A_e * u)(1)
  Eigen::MatrixXd u = generate_random_matrix_mt(mt, n, 1, 0, 1);
  return fabs((A_e1 * u)(0, 0) / (A_e * u)(0, 0));
}

Eigen::MatrixXd
inverse_inferior_triangular (Eigen::MatrixXd A) {
  assert(A.rows() == A.cols());
  int n = A.rows();
  double eps = 1e-5;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i < j)
        assert(fabs(A(i, j)) < eps);
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    B(i, i) = 1 / A(i, i);
    for (int j = i-1; j >= 0; j--) {
      B(i, j) = - (A.row(i).segment(j, i-j) * B.col(j).segment(j, i-j))(0, 0) * B(i, i);
      ///!!segment are al doilea arg lungimea, nu capatul din dreapta
    }
  }
  return B;
}

double calculate_radial_spectrum (std::mt19937 &mt, Eigen::MatrixXd &W) {
  int n = W.cols(); ///!!poate fi apelat si de o matrice (n+1)Xn
  Eigen::MatrixXd N = Eigen::MatrixXd::Zero(n, n), P = N;
  for (int i = 0; i < n; i++)
    N(i, i) = W(i, i);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) P(i, j) = N(i, j) - W(i, j);
  return spectral_radius(mt, inverse_inferior_triangular(N) * P);
}

std::pair<Eigen::MatrixXd, double>
generate_solvable_iterative_matrix
(std::mt19937 &mt, int n, bool type, double coef_a, bool should_calculate_radial_spectrum) {
  ///daca e 1000.0 sunt rezultate wow, nr mai mic de pasi pt hibrid etc
  Eigen::MatrixXd W = generate_random_matrix_mt(mt, n, n, 0, coef_a); ///coef_a: coef marime A

  if (type == true) {
    double coef_jos = 0.001 / coef_a, coef_sus = 1000 * coef_a;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        if (i > j)
          W(i, j) *= coef_jos;
        else if (i < j)
          W(i, j) *= coef_sus;
  }

  Eigen::MatrixXd rand_coef = generate_random_matrix_mt(mt, 1, n, 0, 1); ///poate pui aici > 1

  ///matrice diagonal dominanta
  for (int i = 0; i < n; i++) {
    double oth = W.row(i).sum() - W(i, i);
    W(i, i) = (1 + 0.25 * rand_coef(0, i)) * oth;
  }

  double rad_spectrum = 0;
  if (should_calculate_radial_spectrum)
    rad_spectrum = calculate_radial_spectrum(mt, W);

  return std::make_pair(W, rad_spectrum);
}

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

void
sort_matrix_lines(Eigen::MatrixXd &M,
                  std::function<bool(const Eigen::MatrixXd &, const Eigen::MatrixXd &)> cmp)
{
  int n = M.rows();
  std::vector<Eigen::MatrixXd> linii;
  for (int i = 0; i < n; i++)
    linii.push_back(M.row(i));
  std::stable_sort(linii.begin(), linii.end(), cmp);
  for (int i = 0; i < n; i++)
    M.row(i) = linii[i];
}

///schimba ... intre ele ai ordinea lor sa fie cea din vectorul permutatie dat.
///perm trebuie sa fie vector linie.
///trebuie sa schimb ordinea lui b
///trebuie sa schimb ordinea numerelor de pe ?? lui A
///nu pusesei b.. - 1 .........
void
shift_columns_to_match(Eigen::MatrixXd &A, Eigen::MatrixXd &b, const Eigen::MatrixXd &perm,
                       bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  int n = A.cols();
  std::vector<int> where(n);
  for (int i = 0; i < n; i++)
    where[A(n, i)] = i;

  for (int i = 0; i < n; i++) {
    int tmp = where[perm(0, i)];
    if (tmp != i) {
      where[perm(0, i)] = i;
      where[A(n, i)] = tmp;
      b.row(i).swap(b.row(tmp));
      A.col(i).swap(A.col(tmp));
      A.row(i).swap(A.row(tmp));

      if (should_pass_solution)
        x_precise.row(i).swap(x_precise.row(tmp));
    }
  }
}

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

void
hybrid_parallel_function (Eigen::MatrixXd &A, Eigen::MatrixXd &b, Eigen::MatrixXd &x,
                          Eigen::MatrixXd &x0, int buck_bound, int n, int start_itv, int end_itv)
{
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

void
jacobi_parallel_function (Eigen::MatrixXd &A, Eigen::MatrixXd &b, Eigen::MatrixXd &x,
                          Eigen::MatrixXd &x0, int n, int start_itv, int end_itv)
{
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

  str_print_err += std::to_string(w);

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

std::pair<Eigen::MatrixXd, int>
solve_gauss_seidel_entropy_dp (std::mt19937 &mt, Eigen::MatrixXd A, Eigen::MatrixXd b, double tol,
                               int max_pasi, bool should_pass_solution, Eigen::MatrixXd &x_precise)
{
  if (A.rows() != A.cols() || A.rows() != b.rows() || b.cols() != 1) {
    std::cerr << "solve_gauss_seidel_entropy_dp invalid matrices\n";
    exit(0);
  }

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

  int pasi;
  for (pasi = 1; pasi <= max_pasi; pasi++) {
    Eigen::MatrixXd A_cpy = A, b_cpy = b, x_precise_cpy = x_precise, x_cpy = x;

    ///fac trecere cu sortare pe cpy.
    sort_matrix_lines(x_cpy, []
        (const Eigen::MatrixXd &a_, const Eigen::MatrixXd &b_) {
          return fabs(a_(0, 0) - a_(0, 1)) < fabs(b_(0, 0) - b_(0, 1));
        });

    /// aduc si x_precise_cpy la forma dorita.
    shift_columns_to_match(A_cpy, b_cpy, x_cpy.col(2).transpose(), should_pass_solution, x_precise_cpy);

    ///acum fac trecere directa si pe normal si pe copie.
    x.col(1) = x.col(0);
    x_cpy.col(1) = x_cpy.col(0);
    for (int i = 0; i < n; i++) {
      x(i, 0) = b(i, 0);
      x_cpy(i, 0) = b_cpy(i, 0);
      if (0 <= i) {
        x(i, 0) -= A.row(i).segment(0, i) * x.col(0).segment(0, i);
        x_cpy(i, 0) -= A_cpy.row(i).segment(0, i) * x_cpy.col(0).segment(0, i);
      }
      if (0 <= n-1-i) {
        x(i, 0) -= A.row(i).segment(i+1, n-1-i) * x.col(1).segment(i+1, n-1-i);
        x_cpy(i, 0) -= A_cpy.row(i).segment(i+1, n-1-i) * x_cpy.col(1).segment(i+1, n-1-i);
      }
      x(i, 0) /= A(i, i);
      x_cpy(i, 0) /= A_cpy(i, i);
    }

//    if ((x.col(0) - x_precise).lpNorm<Eigen::Infinity>() >
//        (x_cpy.col(0) - x_precise_cpy).lpNorm<Eigen::Infinity>()) {
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
      print_error_to_solution(pasi, should_pass_solution, "gauss_seidel_entropy_dp_0",
                              x.col(0), x.col(1), x_precise);
    else
      print_error_to_solution(pasi, should_pass_solution, "gauss_seidel_entropy_dp_1",
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

void
one_simulation (std::mt19937 &mt, int n, int nt, double bucketsize_coefficient,
  double coef_a, double coef_b,
  bool should_generate_solution, bool should_calculate_radial_spectrum, bool should_solve_jacobi,
  bool should_solve_gauss_seidel, bool should_solve_jacobi_analytic_parallel,
  bool should_solve_gauss_seidel_analytic, bool should_hybrid_jacobi_gauss_seidel,
  bool should_solve_gauss_seidel_entropy, bool should_hybrid_entropy_jacobi_gauss_seidel,
  bool should_solve_sor_analytic, bool should_solve_gauss_seidel_entropy_dp,
  bool should_hybrid_entropy_dp,
  bool should_pass_solution, bool should_differentiate_generated_matrix_weight)
{
  double bucketsize_coefficient_entropy = 1; //;bucketsize_coefficient * 2 / 3;
  ///se pare ca pentru entropy (dp sau nu) este mai bine sa am 3 galeti in loc de 2.

  std::cout << "Test number " << nt << ", bsz_ct " << bucketsize_coefficient <<\
            ", bsz_ct_entropy " << bucketsize_coefficient_entropy << "\n-------------\n";

  auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();

  auto A_r = generate_solvable_iterative_matrix
             (mt, n, should_differentiate_generated_matrix_weight, coef_a, should_calculate_radial_spectrum);
  ///true = greutate shiftata peste diagonala principala.
  std::cout << "Matricea A are Rho: " << A_r.second << '\n';

  Eigen::MatrixXd A = A_r.first;

  Eigen::MatrixXd b = generate_random_matrix_mt(mt, n, 1, 0, coef_b);

  Eigen::MatrixXd x_precise;

  if (should_generate_solution)
  {
    start = std::chrono::steady_clock::now();
    x_precise = A.colPivHouseholderQr().solve(b);
    stop = std::chrono::steady_clock::now();
    auto duration_hholder = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "timp hholder: " << duration_hholder.count() / 1000000.0 << "s\n-------------\n";
  }
  else
  {
    x_precise = Eigen::MatrixXd::Zero(n, 1);
  }

  if (should_solve_jacobi)
  {
    start = std::chrono::steady_clock::now();
    auto jacobi = solve_jacobi(A, b, 0.00001, 10000, should_pass_solution, x_precise);///penultimul = should_pass_solution

    stop = std::chrono::steady_clock::now();
    auto duration_jacobi = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    Eigen::MatrixXd x_jacobi = jacobi.first;
    int pasi_jacobi = jacobi.second;

    std::cout << (x_jacobi - x_precise).lpNorm<Eigen::Infinity>() << '\n';
    std::cout << "pasi jacobi: " << pasi_jacobi << '\n';
    std::cout << "timp jacobi: " << duration_jacobi.count() / 1000000.0 << "s\n-------------\n";
  }


  if (should_solve_gauss_seidel)
  {
    start = std::chrono::steady_clock::now();
    auto gauss_seidel = solve_gauss_seidel(A, b, 0.00001, 10000, should_pass_solution, x_precise);

    stop = std::chrono::steady_clock::now();
    auto duration_gauss_seidel = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    Eigen::MatrixXd x_gauss_seidel = gauss_seidel.first;
    int pasi_gauss_seidel = gauss_seidel.second;

    std::cout << (x_gauss_seidel - x_precise).lpNorm<Eigen::Infinity>() << '\n';
    std::cout << "pasi gauss_seidel: " << pasi_gauss_seidel << '\n';
    std::cout << "timp gauss_seidel: " << duration_gauss_seidel.count() / 1000000.0 << "s\n-------------\n";
  }


  if (should_solve_jacobi_analytic_parallel)
  {
    start = std::chrono::steady_clock::now();
    auto jacobi_parallel = solve_jacobi_analytic_parallel(A, b, 0.00001, 10000, should_pass_solution, x_precise);

    stop = std::chrono::steady_clock::now();
    auto duration_jacobi_parallel = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    Eigen::MatrixXd x_jacobi_parallel = jacobi_parallel.first;
    int pasi_jacobi_parallel = jacobi_parallel.second;

    std::cout << (x_jacobi_parallel - x_precise).lpNorm<Eigen::Infinity>() << '\n';
    std::cout << "pasi jacobi_parallel: " << pasi_jacobi_parallel << '\n';
    std::cout << "timp jacobi_parallel: " << duration_jacobi_parallel.count() / 1000000.0 << "s\n-------------\n";
  }


  if (should_solve_gauss_seidel_analytic)
  {
    start = std::chrono::steady_clock::now();
    auto gauss_seidel_analytic = solve_gauss_seidel_analytic(A, b, 0.00001, 10000, should_pass_solution, x_precise);

    stop = std::chrono::steady_clock::now();
    auto duration_gauss_seidel_analytic = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    Eigen::MatrixXd x_gauss_seidel_analytic = gauss_seidel_analytic.first;
    int pasi_gauss_seidel_analytic = gauss_seidel_analytic.second;

    std::cout << (x_gauss_seidel_analytic - x_precise).lpNorm<Eigen::Infinity>() << '\n';
    std::cout << "pasi gauss_seidel_analytic: " << pasi_gauss_seidel_analytic << '\n';
    std::cout << "timp gauss_seidel_analytic: " << duration_gauss_seidel_analytic.count() / 1000000.0 << "s\n-------------\n";
  }


  if (should_hybrid_jacobi_gauss_seidel)
  {
    start = std::chrono::steady_clock::now();
    auto hybrid = hybrid_jacobi_gauss_seidel(A, b, 0.00001, 10000, bucketsize_coefficient, should_pass_solution, x_precise);

    stop = std::chrono::steady_clock::now();
    auto duration_hybrid = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    Eigen::MatrixXd x_hybrid = hybrid.first;
    int pasi_hybrid = hybrid.second;

    std::cout << (x_hybrid - x_precise).lpNorm<Eigen::Infinity>() << '\n';
    std::cout << "pasi hibrid: " << pasi_hybrid << '\n';
    std::cout << "timp hibrid: " << duration_hybrid.count() / 1000000.0 << "s\n-------------\n";
  }

  if (should_solve_gauss_seidel_entropy)
  {
    start = std::chrono::steady_clock::now();
    auto gs_entropy = solve_gauss_seidel_entropy(mt, A, b, 0.00001, 100, should_pass_solution, x_precise);

    stop = std::chrono::steady_clock::now();
    auto duration_gs_entropy = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    Eigen::MatrixXd x_gs_entropy = gs_entropy.first;
    int pasi_gs_entropy = gs_entropy.second;

    std::cout << (x_gs_entropy - x_precise).lpNorm<Eigen::Infinity>() << '\n';
    std::cout << "pasi gs entropy: " << pasi_gs_entropy << '\n';
    std::cout << "timp gs entropy: " << duration_gs_entropy.count() / 1000000.0 << "s\n-------------\n";
  }

  if (should_hybrid_entropy_jacobi_gauss_seidel)
  {
    start = std::chrono::steady_clock::now();
    auto hybrid_entropy = hybrid_entropy_jacobi_gauss_seidel
                          (mt, A, b, 0.00001, 100, bucketsize_coefficient_entropy,
                           should_pass_solution, x_precise);
    /// in loc de 1 nu ar trebui bucketsize_coefficient?? ca sa am doar 2 galeti, nu sqrt(n)
    /// bucketsize_coefficient da mai rau decat 1.

    stop = std::chrono::steady_clock::now();
    auto duration_hybrid_entropy = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    Eigen::MatrixXd x_hybrid_entropy = hybrid_entropy.first;
    int pasi_hybrid_entropy = hybrid_entropy.second;

    std::cout << (x_hybrid_entropy - x_precise).lpNorm<Eigen::Infinity>() << '\n';
    std::cout << "pasi hibrid entropy: " << pasi_hybrid_entropy << '\n';
    std::cout << "timp hibrid entropy: " << duration_hybrid_entropy.count() / 1000000.0 << "s\n-------------\n";
  }

  if (should_solve_sor_analytic)
  {
    start = std::chrono::steady_clock::now();

    auto sor_analytic = solve_sor_analytic(mt, A, b, 0.00001, 100, "solve_sor_analytic_",
                                           should_pass_solution, x_precise);

    stop = std::chrono::steady_clock::now();
    auto duration_sor_analytic = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    Eigen::MatrixXd x_sor_analytic = sor_analytic.first;
    int pasi_sor_analytic = sor_analytic.second;

    std::cout << (x_sor_analytic - x_precise).lpNorm<Eigen::Infinity>() << '\n';
    std::cout << "pasi sor analytic: " << pasi_sor_analytic << '\n';
    std::cout << "timp sor analytic: " << duration_sor_analytic.count() / 1000000.0 << "s\n-------------\n";
  }

  if (should_solve_gauss_seidel_entropy_dp)
  {
    start = std::chrono::steady_clock::now();
    auto gs_entropy_dp = solve_gauss_seidel_entropy_dp(mt, A, b, 0.00001, 100, should_pass_solution, x_precise);

    stop = std::chrono::steady_clock::now();
    auto duration_gs_entropy_dp = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    Eigen::MatrixXd x_gs_entropy_dp = gs_entropy_dp.first;
    int pasi_gs_entropy_dp = gs_entropy_dp.second;

    std::cout << (x_gs_entropy_dp - x_precise).lpNorm<Eigen::Infinity>() << '\n';
    std::cout << "pasi gauss_seidel_entropy_dp: " << pasi_gs_entropy_dp << '\n';
    std::cout << "timp gauss_seidel_entropy_dp: " << duration_gs_entropy_dp.count() / 1000000.0 << "s\n-------------\n";
  }

  if (should_hybrid_entropy_dp)
  {
    start = std::chrono::steady_clock::now();
    auto pair_hybrid_entropy_dp = hybrid_entropy_dp(mt, A, b, 0.00001, 100, bucketsize_coefficient_entropy,
                                                    should_pass_solution, x_precise);

    stop = std::chrono::steady_clock::now();
    auto duration_hybrid_entropy_dp = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    Eigen::MatrixXd x_hybrid_entropy_dp = pair_hybrid_entropy_dp.first;
    int pasi_hybrid_entropy_dp = pair_hybrid_entropy_dp.second;

    std::cout << (x_hybrid_entropy_dp - x_precise).lpNorm<Eigen::Infinity>() << '\n';
    std::cout << "pasi hybrid_entropy_dp: " << pasi_hybrid_entropy_dp << '\n';
    std::cout << "timp hybrid_entropy_dp: " << duration_hybrid_entropy_dp.count() / 1000000.0 << "s\n-------------\n";
  }
}

void
nonentropy_stresstest (std::mt19937 &mt)
{
  bool should_generate_solution = false,
       should_calculate_radial_spectrum = false,
       should_solve_jacobi = false,
       should_solve_gauss_seidel = false,
       should_solve_jacobi_analytic_parallel = true,
       should_solve_gauss_seidel_analytic = true,
       should_hybrid_jacobi_gauss_seidel = true,
       should_solve_gauss_seidel_entropy = false,
       should_hybrid_entropy_jacobi_gauss_seidel = false,
       should_solve_sor_analytic = false,
       should_solve_gauss_seidel_entropy_dp = false,
       should_hybrid_entropy_dp = false,
       should_pass_solution = false;

  int n, num_tests[8] = {0};

  std::cout << "NUMBER OF THREADS: " << std::thread::hardware_concurrency() << '\n' << std::flush;

  std::cout << "Matrix dimension: " << std::flush;
  std::cin >> n;
  std::cout << "Number of tests (variating bucketsize coefficient, a_coef 1, b_coef 1, eq weight): " << std::flush;
  std::cin >> num_tests[0];
  std::cout << "Number of tests (variating bucketsize coefficient, a_coef 1000, b_coef 1000, eq weight): " << std::flush;
  std::cin >> num_tests[1];
  std::cout << "Number of tests (variating bucketsize coefficient, a_coef 1, b_coef 1, diff weight): " << std::flush;
  std::cin >> num_tests[2];
  std::cout << "Number of tests (variating bucketsize coefficient, a_coef 1000, b_coef 1000, diff weight): " << std::flush;
  std::cin >> num_tests[3];

  std::cout << "Number of tests (fixed bucketsize coefficient = sqrt(n)/2, a_coef 1, b_coef 1, eq weight): " << std::flush;
  std::cin >> num_tests[4];
  std::cout << "Number of tests (fixed bucketsize coefficient = sqrt(n)/2, a_coef 1000, b_coef 1000, eq weight): " << std::flush;
  std::cin >> num_tests[5];
  std::cout << "Number of tests (fixed bucketsize coefficient = sqrt(n)/2, a_coef 1, b_coef 1, diff weight): " << std::flush;
  std::cin >> num_tests[6];
  std::cout << "Number of tests (fixed bucketsize coefficient = sqrt(n)/2, a_coef 1000, b_coef 1000, diff weight): " << std::flush;
  std::cin >> num_tests[7];

  std::uniform_int_distribution<int> distr_int(0, 100);

  std::cout << "\n\n(variating bucketsize coefficient, a_coef 1, b_coef 1, eq weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[0]; nt++) {
    one_simulation(mt, n, nt, distr_int(mt), 1.0, 1.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, false);

    std::cout << std::flush;
  }

  std::cout << "\n\n(variating bucketsize coefficient, a_coef 1000, b_coef 1000, eq weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[1]; nt++) {
    one_simulation(mt, n, nt, distr_int(mt), 1000.0, 1000.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, false);

    std::cout << std::flush;
  }

  std::cout << "\n\n(variating bucketsize coefficient, a_coef 1, b_coef 1, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[2]; nt++) {
    one_simulation(mt, n, nt, distr_int(mt), 1.0, 1.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, true);

    std::cout << std::flush;
  }

  std::cout << "\n\n(variating bucketsize coefficient, a_coef 1000, b_coef 1000, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[3]; nt++) {
    one_simulation(mt, n, nt, distr_int(mt), 1000.0, 1000.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, true);

    std::cout << std::flush;
  }



  std::cout << "\n\n(fixed bucketsize coefficient = sqrt(n)/2, a_coef 1, b_coef 1, eq weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[4]; nt++) {
    one_simulation(mt, n, nt, sqrt(n) / 2 + 1, 1.0, 1.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, false);

    std::cout << std::flush;
  }

  std::cout << "\n\n(fixed bucketsize coefficient = sqrt(n)/2, a_coef 1000, b_coef 1000, eq weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[5]; nt++) {
    one_simulation(mt, n, nt, sqrt(n) / 2 + 1, 1000.0, 1000.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, false);

    std::cout << std::flush;
  }

  std::cout << "\n\n(fixed bucketsize coefficient = sqrt(n)/2, a_coef 1, b_coef 1, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[6]; nt++) {
    one_simulation(mt, n, nt, sqrt(n) / 2 + 1, 1.0, 1.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, true);

    std::cout << std::flush;
  }

  std::cout << "\n\n(fixed bucketsize coefficient = sqrt(n)/2, a_coef 1000, b_coef 1000, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[7]; nt++) {
    one_simulation(mt, n, nt, sqrt(n) / 2 + 1, 1000.0, 1000.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, true);

    std::cout << std::flush;
  }
}

void
entropy_convergencetest (std::mt19937 &mt)
{
  bool should_generate_solution = true,
       should_calculate_radial_spectrum = false,
       should_solve_jacobi = false,
       should_solve_gauss_seidel = false,
       should_solve_jacobi_analytic_parallel = true,
       should_solve_gauss_seidel_analytic = true,
       should_hybrid_jacobi_gauss_seidel = true,
       should_solve_gauss_seidel_entropy = true,
       should_hybrid_entropy_jacobi_gauss_seidel = true,
       should_solve_sor_analytic = true,
       should_solve_gauss_seidel_entropy_dp = true,
       should_hybrid_entropy_dp = true,
       should_pass_solution = true,
       should_differentiate_generated_matrix_weight = true;

  std::cout << "NUMBER OF THREADS: " << std::thread::hardware_concurrency() << '\n' << std::flush;

  int n;
  std::cout << "Matrix dimension: " << std::flush;
  std::cin >> n;

  std::cout << "fixed bucketsize coefficient = sqrt(n)/2+1, a_coef 1000, b_coef 1000, diff weight):\n" << std::flush;
  ///incearca si norma 2 in loc de inf?

  one_simulation(mt, n, 1, sqrt(n) / 2 + 1, 1000.0, 1000.0,
  should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
  should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
  should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
  should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
  should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
  should_hybrid_entropy_dp,
  should_pass_solution, should_differentiate_generated_matrix_weight);
}

void
dp_stresstest (std::mt19937 &mt)
{
  bool should_generate_solution = false,
       should_calculate_radial_spectrum = false,
       should_solve_jacobi = false,
       should_solve_gauss_seidel = false,
       should_solve_jacobi_analytic_parallel = true,
       should_solve_gauss_seidel_analytic = true,
       should_hybrid_jacobi_gauss_seidel = true,
       should_solve_gauss_seidel_entropy = true,
       should_hybrid_entropy_jacobi_gauss_seidel = true,
       should_solve_sor_analytic = false,
       should_solve_gauss_seidel_entropy_dp = true,
       should_hybrid_entropy_dp = true,
       should_pass_solution = false;

  int n, num_tests[8] = {0};

  std::cout << "NUMBER OF THREADS: " << std::thread::hardware_concurrency() << '\n' << std::flush;

  std::cout << "Matrix dimension: " << std::flush;
  std::cin >> n;
  std::cout << "Number of tests (variating bucketsize coefficient, a_coef 1000, b_coef 1000, diff weight): " << std::flush;
  std::cin >> num_tests[0];

  std::uniform_int_distribution<int> distr_int(1, 100);

  std::cout << "\n\n(variating bucketsize coefficient, a_coef 1000, b_coef 1000, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[0]; nt++) {
    one_simulation(mt, n, nt, distr_int(mt), 1000.0, 1000.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, true); /// ultimul parametru este should_differentiate_generated_matrix_weight

    std::cout << std::flush;
  }
}

int
main()
{
  Eigen::initParallel();
  ///nu ajuta pt versiunea asta Eigen, dar ar tb pus pt ca folosesc si eu op mele in paralel
  std::mt19937 mt = std::mt19937();
  mt.seed(time(NULL));

  //nonentropy_stresstest(mt);
  entropy_convergencetest(mt);
  //dp_stresstest(mt);

  return 0;
}
