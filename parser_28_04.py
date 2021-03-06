fin = open("log_5000_timpi_hgse_gsedp_hgsedp.txt", "r")
lines = fin.read().split('\n')

bsz_coef = []

pasi_jacobi = []
pasi_gauss_seidel = []
pasi_jacobi_parallel = []
pasi_gauss_seidel_analytic = []
pasi_hibrid = []
pasi_hibrid_entropy = []
pasi_sor = []
pasi_gauss_seidel_entropy_dp = []
pasi_hybrid_entropy_dp = []

timp_jacobi = []
timp_gauss_seidel = []
timp_jacobi_parallel = []
timp_gauss_seidel_analytic = []
timp_hibrid = []
timp_hibrid_entropy = []
timp_sor = []
timp_gauss_seidel_entropy_dp = []
timp_hybrid_entropy_dp = []

for s in lines:
    w = s.split(' ')
    if len(w) > 0 and w[0] == "Test":
        bsz_coef.append(float(w[4]))
    if len(w) > 2 and w[0] == "pasi":
        if w[1] == "jacobi:":
            pasi_jacobi.append(int(w[2]))
        elif w[1] == "gauss_seidel:":
            pasi_gauss_seidel.append(int(w[2]))
        elif w[1] == "jacobi_parallel:":
            pasi_jacobi_parallel.append(int(w[2]))
        elif w[1] == "gauss_seidel_analytic:":
            pasi_gauss_seidel_analytic.append(int(w[2]))
        elif w[1] == "hibrid:":
            pasi_hibrid.append(int(w[2]))
        elif w[1] == "hibrid":
            pasi_hibrid_entropy.append(int(w[3]))
        elif w[1] == "gauss_seidel_entropy_dp:":
            pasi_gauss_seidel_entropy_dp.append(int(w[2]))
        elif w[1] == "hybrid_entropy_dp:":
            pasi_hybrid_entropy_dp.append(int(w[2]))
            
    elif len(w) > 2 and w[0] == "timp":
        if w[1] == "jacobi:":
            timp_jacobi.append(float(w[2].split('s')[0]))
        elif w[1] == "gauss_seidel:":
            timp_gauss_seidel.append(float(w[2].split('s')[0]))
        elif w[1] == "jacobi_parallel:":
            timp_jacobi_parallel.append(float(w[2].split('s')[0]))
        elif w[1] == "gauss_seidel_analytic:":
            timp_gauss_seidel_analytic.append(float(w[2].split('s')[0]))
        elif w[1] == "hibrid:":
            timp_hibrid.append(float(w[2].split('s')[0]))
        elif w[1] == "hibrid":
            timp_hibrid_entropy.append(float(w[3].split('s')[0]))
        elif w[1] == "gauss_seidel_entropy_dp:":
            timp_gauss_seidel_entropy_dp.append(float(w[2].split('s')[0]))
        elif w[1] == "hybrid_entropy_dp:":
            timp_hybrid_entropy_dp.append(float(w[2].split('s')[0]))
        
print("bsz_coef = ", bsz_coef, ";")

print("pasi_jacobi = ", pasi_jacobi, ";")
print("pasi_gauss_seidel = ", pasi_gauss_seidel, ";")
print("pasi_jacobi_parallel = ", pasi_jacobi_parallel, ";")
print("pasi_gauss_seidel_analytic = ", pasi_gauss_seidel_analytic, ";")
print("pasi_hibrid = ", pasi_hibrid, ";")
print("pasi_hibrid_entropy = ", pasi_hibrid_entropy, ";")
print("pasi_gauss_seidel_entropy_dp = ", pasi_gauss_seidel_entropy_dp, ";")
print("pasi_hybrid_entropy_dp = ", pasi_hybrid_entropy_dp, ";")

print("timp_jacobi = ", timp_jacobi, ";")
print("timp_gauss_seidel = ", timp_gauss_seidel, ";")
print("timp_jacobi_parallel = ", timp_jacobi_parallel, ";")
print("timp_gauss_seidel_analytic = ", timp_gauss_seidel_analytic, ";")
print("timp_hibrid = ", timp_hibrid, ";")
print("timp_hibrid_entropy = ", timp_hibrid_entropy, ";")
print("timp_gauss_seidel_entropy_dp = ", timp_gauss_seidel_entropy_dp, ";")
print("timp_hybrid_entropy_dp = ", timp_hybrid_entropy_dp, ";")

fin.close()
