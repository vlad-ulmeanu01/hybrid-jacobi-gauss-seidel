fin = open("log_5000_j_p_gs_a_gse_hgse_ff_interesant_6.txt", "r")
lines = fin.read().split('\n')

pasi_hibrid = []
pasi_hibrid_entropy = []
pasi_gauss_seidel_entropy = []
pasi_jacobi_parallel = []
pasi_gauss_seidel_analytic = []

err_sol_hibrid = []
err_sol_hibrid_entropy = []
err_sol_gauss_seidel_entropy = []
err_sol_jacobi_parallel = []
err_sol_gauss_seidel_analytic = []

err_rel_hibrid = []
err_rel_hibrid_entropy = []
err_rel_gauss_seidel_entropy = []
err_rel_jacobi_parallel = []
err_rel_gauss_seidel_analytic = []

for s in lines:
    w = s.split(' ')
    if len(w) >= 4 and w[1] == "pasi;" and w[2] == "err_fata_de_solutie":
        if w[3] == "hibrid:":
            pasi_hibrid.append(int(w[0]))
        elif w[3] == "hibrid_entropy:":
            pasi_hibrid_entropy.append(int(w[0]))
        elif w[3] == "gauss_seidel_entropy:":
            pasi_gauss_seidel_entropy.append(int(w[0]))
        elif w[3] == "jacobi_parallel:":
            pasi_jacobi_parallel.append(int(w[0]))
        elif w[3] == "gauss_seidel_analytic:":
            pasi_gauss_seidel_analytic.append(int(w[0]))

    if len(w) >= 5 and w[3] == "hibrid:":
        if w[2] == "err_fata_de_solutie":
            err_sol_hibrid.append(float(w[4]))
        if w[2] == "err_fata_de_ult_val":
            err_rel_hibrid.append(float(w[4]))
    
    if len(w) >= 5 and w[3] == "hibrid_entropy:":
        if w[2] == "err_fata_de_solutie":
            err_sol_hibrid_entropy.append(float(w[4]))
        if w[2] == "err_fata_de_ult_val":
            err_rel_hibrid_entropy.append(float(w[4]))

    if len(w) >= 5 and w[3] == "gauss_seidel_entropy:":
        if w[2] == "err_fata_de_solutie":
            err_sol_gauss_seidel_entropy.append(float(w[4]))
        if w[2] == "err_fata_de_ult_val":
            err_rel_gauss_seidel_entropy.append(float(w[4]))

    if len(w) >= 5 and w[3] == "jacobi_parallel:":
        if w[2] == "err_fata_de_solutie":
            err_sol_jacobi_parallel.append(float(w[4]))
        if w[2] == "err_fata_de_ult_val":
            err_rel_jacobi_parallel.append(float(w[4]))

    if len(w) >= 5 and w[3] == "gauss_seidel_analytic:":
        if w[2] == "err_fata_de_solutie":
            err_sol_gauss_seidel_analytic.append(float(w[4]))
        if w[2] == "err_fata_de_ult_val":
            err_rel_gauss_seidel_analytic.append(float(w[4]))

print("pasi_hibrid = ", pasi_hibrid, ";")
print("pasi_hibrid_entropy = ", pasi_hibrid_entropy, ";")
print("pasi_gauss_seidel_entropy = ", pasi_gauss_seidel_entropy, ";")
print("pasi_jacobi_parallel = ", pasi_jacobi_parallel, ";")
print("pasi_gauss_seidel_analytic = ", pasi_gauss_seidel_analytic, ";")

print("err_sol_hibrid = ", err_sol_hibrid, ";")
print("err_sol_hibrid_entropy = ", err_sol_hibrid_entropy, ";")
print("err_sol_gauss_seidel_entropy = ", err_sol_gauss_seidel_entropy, ";")
print("err_sol_jacobi_parallel = ", err_sol_jacobi_parallel, ";")
print("err_sol_gauss_seidel_analytic = ", err_sol_gauss_seidel_analytic, ";")

print("err_rel_hibrid = ", err_rel_hibrid, ";")
print("err_rel_hibrid_entropy = ", err_rel_hibrid_entropy, ";")
print("err_rel_gauss_seidel_entropy = ", err_rel_gauss_seidel_entropy, ";")
print("err_rel_jacobi_parallel = ", err_rel_jacobi_parallel, ";")
print("err_rel_gauss_seidel_analytic = ", err_rel_gauss_seidel_analytic, ";")

fin.close()
