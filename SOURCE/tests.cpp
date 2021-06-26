#include "tests.h"

void
one_simulation (std::mt19937 &mt, int n, int nt, double bucketsize_coefficient,
  double coef_a, double coef_b, double coef_sparse,
  bool should_generate_solution, bool should_calculate_radial_spectrum, bool should_solve_jacobi,
  bool should_solve_gauss_seidel, bool should_solve_jacobi_analytic_parallel,
  bool should_solve_gauss_seidel_analytic, bool should_hybrid_jacobi_gauss_seidel,
  bool should_solve_gauss_seidel_entropy, bool should_hybrid_entropy_jacobi_gauss_seidel,
  bool should_solve_sor_analytic, bool should_solve_gauss_seidel_entropy_dp,
  bool should_hybrid_entropy_dp,
  bool should_pass_solution, bool is_system_dense, bool should_differentiate_generated_matrix_weight)
{
  double bucketsize_coefficient_entropy = 1; //;bucketsize_coefficient * 2 / 3;
  ///se pare ca pentru entropy (dp sau nu) este mai bine sa am 3 galeti in loc de 2.

  std::cout << "Test number " << nt << ", bsz_ct " << bucketsize_coefficient <<\
            ", bsz_ct_entropy " << bucketsize_coefficient_entropy << "\n-------------\n";

  auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n), b = Eigen::MatrixXd::Zero(n, 1), x_precise = Eigen::MatrixXd::Zero(n, 1);
  if (is_system_dense)
  {
    auto A_r = generate_solvable_iterative_matrix
               (mt, n, should_differentiate_generated_matrix_weight, coef_a, should_calculate_radial_spectrum);
    ///true = greutate shiftata peste diagonala principala.
    std::cout << "Matricea A are Rho: " << A_r.second << '\n';

    A = A_r.first;
    b = generate_random_matrix_mt(mt, n, 1, 0, coef_b);

    if (should_generate_solution) {
      start = std::chrono::steady_clock::now();
      x_precise = A.colPivHouseholderQr().solve(b);
      stop = std::chrono::steady_clock::now();
      auto duration_hholder = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      std::cout << "timp hholder: " << duration_hholder.count() / 1000000.0 << "s\n-------------\n";
    } else
      x_precise = Eigen::MatrixXd::Zero(n, 1);
  }
  else
  {
    populate_sparse_matrix(mt, A, x_precise, b, coef_sparse, 0, 1000);
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
       should_pass_solution = false,
       is_system_dense = true;

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
    one_simulation(mt, n, nt, distr_int(mt), 1.0, 1.0, 0.0, /// <- ult este coef_sparse, nu ai treaba aici.
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, is_system_dense, false);

    std::cout << std::flush;
  }

  std::cout << "\n\n(variating bucketsize coefficient, a_coef 1000, b_coef 1000, eq weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[1]; nt++) {
    one_simulation(mt, n, nt, distr_int(mt), 1000.0, 1000.0, 0.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, is_system_dense, false);

    std::cout << std::flush;
  }

  std::cout << "\n\n(variating bucketsize coefficient, a_coef 1, b_coef 1, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[2]; nt++) {
    one_simulation(mt, n, nt, distr_int(mt), 1.0, 1.0, 0.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, is_system_dense, true);

    std::cout << std::flush;
  }

  std::cout << "\n\n(variating bucketsize coefficient, a_coef 1000, b_coef 1000, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[3]; nt++) {
    one_simulation(mt, n, nt, distr_int(mt), 1000.0, 1000.0, 0.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, is_system_dense, true);

    std::cout << std::flush;
  }



  std::cout << "\n\n(fixed bucketsize coefficient = sqrt(n)/2, a_coef 1, b_coef 1, eq weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[4]; nt++) {
    one_simulation(mt, n, nt, sqrt(n) / 2 + 1, 1.0, 1.0, 0.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, is_system_dense, false);

    std::cout << std::flush;
  }

  std::cout << "\n\n(fixed bucketsize coefficient = sqrt(n)/2, a_coef 1000, b_coef 1000, eq weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[5]; nt++) {
    one_simulation(mt, n, nt, sqrt(n) / 2 + 1, 1000.0, 1000.0, 0.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, is_system_dense, false);

    std::cout << std::flush;
  }

  std::cout << "\n\n(fixed bucketsize coefficient = sqrt(n)/2, a_coef 1, b_coef 1, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[6]; nt++) {
    one_simulation(mt, n, nt, sqrt(n) / 2 + 1, 1.0, 1.0, 0.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, is_system_dense, true);

    std::cout << std::flush;
  }

  std::cout << "\n\n(fixed bucketsize coefficient = sqrt(n)/2, a_coef 1000, b_coef 1000, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[7]; nt++) {
    one_simulation(mt, n, nt, sqrt(n) / 2 + 1, 1000.0, 1000.0, 0.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, is_system_dense, true);

    std::cout << std::flush;
  }
}

void
entropy_convergencetest (std::mt19937 &mt)
{
  bool should_generate_solution = false,  ///T
       should_calculate_radial_spectrum = false,
       should_solve_jacobi = false,
       should_solve_gauss_seidel = false,
       should_solve_jacobi_analytic_parallel = false,  ///T
       should_solve_gauss_seidel_analytic = false,  ///T
       should_hybrid_jacobi_gauss_seidel = true,
       should_solve_gauss_seidel_entropy = true,
       should_hybrid_entropy_jacobi_gauss_seidel = true,
       should_solve_sor_analytic = false,  ///T
       should_solve_gauss_seidel_entropy_dp = true,
       should_hybrid_entropy_dp = true,
       should_pass_solution = false,  ///T
       is_system_dense = true,
       should_differentiate_generated_matrix_weight = true;

  std::cout << "NUMBER OF THREADS: " << std::thread::hardware_concurrency() << '\n' << std::flush;

  int n;
  std::cout << "Matrix dimension: " << std::flush;
  std::cin >> n;

  std::cout << "fixed bucketsize coefficient = sqrt(n)/2+1, a_coef 1000, b_coef 1000, diff weight):\n" << std::flush;
  ///incearca si norma 2 in loc de inf?

  one_simulation(mt, n, 1, sqrt(n) / 2 + 1, 1000.0, 1000.0, 0.0,
  should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
  should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
  should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
  should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
  should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
  should_hybrid_entropy_dp,
  should_pass_solution, is_system_dense, should_differentiate_generated_matrix_weight);
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
       should_pass_solution = false,
       is_system_dense = true;

  int n, num_tests[8] = {0};

  std::cout << "NUMBER OF THREADS: " << std::thread::hardware_concurrency() << '\n' << std::flush;

  std::cout << "Matrix dimension: " << std::flush;
  std::cin >> n;
  std::cout << "Number of tests (variating bucketsize coefficient, a_coef 1000, b_coef 1000, diff weight): " << std::flush;
  std::cin >> num_tests[0];

  std::uniform_int_distribution<int> distr_int(1, 100);

  std::cout << "\n\n(variating bucketsize coefficient, a_coef 1000, b_coef 1000, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[0]; nt++) {
    one_simulation(mt, n, nt, distr_int(mt), 1000.0, 1000.0, 0.0,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, is_system_dense, true); /// ultimul parametru este should_differentiate_generated_matrix_weight

    std::cout << std::flush;
  }
}

void
small_sparse (std::mt19937 &mt)
{
  int n = 10;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n), b = Eigen::MatrixXd::Zero(n, 1), x_precise = Eigen::MatrixXd::Zero(n, 1);
  double tol = 1e-5;

  A << 4, -1, -1,  0,  0,  0,  0,  0,  0,  0,
      -1,  5, -1, -1, -1,  0,  0,  0,  0,  0,
      -1, -1,  5,  0, -1, -1,  0,  0,  0,  0,
       0, -1,  0,  5, -1,  0, -1, -1,  0,  0,
       0, -1, -1, -1,  6, -1,  0, -1, -1,  0,
       0,  0, -1,  0, -1,  5,  0,  0, -1, -1,
       0,  0,  0, -1,  0,  0,  4, -1,  0,  0,
       0,  0,  0, -1, -1,  0, -1,  5, -1,  0,
       0,  0,  0,  0, -1, -1,  0, -1,  5, -1,
       0,  0,  0,  0,  0, -1,  0,  0, -1,  4;

  b << 0, 0, 0, 0, 0, 0, 1, 1, 1, 1;

  auto pair_jacobi_parallel = solve_jacobi_analytic_parallel(A, b, tol, 100, false, x_precise);
  std::cout << "jacobi parallel " << pair_jacobi_parallel.second << '\n';

  auto pair_gauss_seidel = solve_gauss_seidel_analytic(A, b, tol, 100, false, x_precise);
  std::cout << "gauss seidel analytic " << pair_gauss_seidel.second << '\n';

  auto pair_hybrid = hybrid_jacobi_gauss_seidel(A, b, tol, 100, sqrt(n)/2+1, false, x_precise);
  std::cout << "hybrid " << pair_hybrid.second << '\n';

  auto pair_gse_dp = solve_gauss_seidel_entropy_dp(mt, A, b, tol, 100, true, x_precise);
  std::cout << "entropy dp " << pair_gse_dp.second << '\n';

  auto pair_hgse_dp = hybrid_entropy_dp(mt, A, b, tol, 100, 1, false, x_precise);
  std::cout << "hybrid entropy dp " << pair_hgse_dp.second << '\n';

  std::cout << pair_gse_dp.first << '\n';
}

void
sparse_stresstest (std::mt19937 &mt)
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
       should_solve_sor_analytic = true,
       should_solve_gauss_seidel_entropy_dp = true,
       should_hybrid_entropy_dp = true,
       should_pass_solution = true,
       is_system_dense = false;

  int n, num_tests[8] = {0};
  double coef_sparse = 0.0;

  std::cout << "NUMBER OF THREADS: " << std::thread::hardware_concurrency() << '\n' << std::flush;

  std::cout << "Matrix dimension: " << std::flush;
  std::cin >> n;
  std::cout << "Percentage in [0, 1] of nonzero elements in sparse matrix: " << std::flush;
  std::cin >> coef_sparse;
  std::cout << "Number of tests (sparse system, fixed bucketsize coefficient, a_coef 1000, b_coef 1000, diff weight): " << std::flush;
  std::cin >> num_tests[0];

  std::cout << "\n\n(variating bucketsize coefficient, a_coef 1000, b_coef 1000, diff weight):\n" << std::flush;
  for (int nt = 1; nt <= num_tests[0]; nt++) {
    one_simulation(mt, n, nt, sqrt(n) / 2 + 1, 1000.0, 1000.0, coef_sparse,
    should_generate_solution, should_calculate_radial_spectrum, should_solve_jacobi,
    should_solve_gauss_seidel, should_solve_jacobi_analytic_parallel,
    should_solve_gauss_seidel_analytic, should_hybrid_jacobi_gauss_seidel,
    should_solve_gauss_seidel_entropy, should_hybrid_entropy_jacobi_gauss_seidel,
    should_solve_sor_analytic, should_solve_gauss_seidel_entropy_dp,
    should_hybrid_entropy_dp,
    should_pass_solution, is_system_dense, true); /// ultimul parametru este should_differentiate_generated_matrix_weight

    std::cout << std::flush;
  }
}
