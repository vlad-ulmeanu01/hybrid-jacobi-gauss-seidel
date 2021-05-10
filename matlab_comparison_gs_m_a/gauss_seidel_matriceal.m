function [x, pasi, timp] = gauss_seidel_matriceal (A, b, err)
  tic;
  [x, pasi] = deal(zeros(length(A), 1), 0);
  N = tril(A);
  P = N - A;
  [G, c] = deal(N \ P, N \ b);
  %inv_N = inv(N);
  %inv_N = inverse_inferior_triangular(N);
  %[G, c] = deal(inv_N * P, inv_N * b);
  while pasi == 0 || max(abs(x - x_last)) > err
    x_last = x;
    x = G * x_last + c;
    pasi = pasi + 1;
  end
  timp = toc;
end