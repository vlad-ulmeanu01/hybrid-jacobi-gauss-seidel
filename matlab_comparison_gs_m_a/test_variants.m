function [] = test_variants(n)
    [A, b, rs] = generate_matrix_subunitary_radial_spectrum(n);
    fprintf("Rho: %f\n", rs);
    x_precise = A \ b;
    [x, pasi, timp] = gauss_seidel_analytic(A, b, 1e-5);
    [x_m, pasi_m, timp_m] = gauss_seidel_matriceal(A, b, 1e-5);
    fprintf("timp gauss seidel analytic %f, %d pasi, err %f\n", timp, pasi, max(abs(x - x_precise)));
    fprintf("timp gauss seidel matriceal %f, %d pasi, err %f\n", timp_m, pasi_m, max(abs(x_m - x_precise)));
end