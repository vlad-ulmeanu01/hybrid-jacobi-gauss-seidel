function [W, b, rad_spectrum] = generate_matrix_subunitary_radial_spectrum (n)
    %vreau sa fac o matrice diagonal superior dominanta
    W = rand(n) * 1000;
    for i = 1:n
        for j = 1:n
            if i > j
               W(i, j) = W(i, j) / 1000000; 
            end
            if i < j
                W(i, j) = W(i, j) * 1000000;
            end
        end
    end
    for p = 1:n
        oth = sum(W(p, :)) - W(p, p);
        W(p, p) = (1 + 0.25 * rand) * oth; %Rho se schimba daca modificam coeficientul lui rand
    end
    b = rand(n, 1) * 1000;
    rad_spectrum = 0;
    %N = diag(diag(W));
    %P = N - W;
    %rad_spectrum = max(abs(eig(inv(N) * P)));
end