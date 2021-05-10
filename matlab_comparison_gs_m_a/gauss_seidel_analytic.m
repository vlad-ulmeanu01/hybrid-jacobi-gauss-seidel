function [x, pasi, timp] = gauss_seidel_analytic (A, b, err)
  tic;
  n = length(A);
  x = zeros(n, 1);
  pasi = 0;
  while pasi == 0 || max(abs(x - x_last)) > err
    x_last = x;
    for i = 1:n
      x(i) = b(i) - A(i, 1:i-1) * x(1:i-1) - A(i, i+1:n) * x_last(i+1:n);
      x(i) = x(i) / A(i, i);
    end
    pasi = pasi + 1;
  end
  timp = toc;
end