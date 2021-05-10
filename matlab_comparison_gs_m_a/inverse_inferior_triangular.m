function [inv_A] = inverse_inferior_triangular (A)
  n = length(A);
  inv_A = zeros(n, n);

  for i = 1:n
    inv_A(i, i) = 1 / A(i, i);
    for j = i-1:-1:1
      inv_A(i, j) = - A(i, j:i-1) * inv_A(j:i-1, j) * inv_A(i, i);
    end
  end
end