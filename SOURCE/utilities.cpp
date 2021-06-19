#include "utilities.h"

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

Eigen::MatrixXd
generate_random_matrix_mt (std::mt19937 &mt, int n, int m, double mi, double Ma)
{
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
inverse_inferior_triangular (Eigen::MatrixXd A)
{
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

double
calculate_radial_spectrum (std::mt19937 &mt, Eigen::MatrixXd &W)
{
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
(std::mt19937 &mt, int n, bool type, double coef_a, bool should_calculate_radial_spectrum)
{
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

  std::vector<std::pair<int, int> > delayed_row_swaps;

  for (int i = 0; i < n; i++) {
    int tmp = where[perm(0, i)];
    if (tmp != i) {
      where[perm(0, i)] = i;
      where[A(n, i)] = tmp;
      b.row(i).swap(b.row(tmp));

      /// ca sa mentin ordinea pe linia cu permutarile; am nevoie sa o tin actualizata aici pentru where
      /// de aia mergea doar delayed_row_swaps si nu ar fi mers delayed_column_swaps
      A.col(i).swap(A.col(tmp));
      //A.row(i).swap(A.row(tmp));
      delayed_row_swaps.push_back(std::make_pair(i, tmp));

      if (should_pass_solution)
        x_precise.row(i).swap(x_precise.row(tmp));
    }
  }

  for (std::pair<int, int> p: delayed_row_swaps)
    A.row(p.first).swap(A.row(p.second));
}
