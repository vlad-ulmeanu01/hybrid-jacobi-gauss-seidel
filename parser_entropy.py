fin = open("loguri_13_06/log_5000_he_gsedp_sor_hedp.txt", "r")
lines = fin.read().split('\n')

pasi_hibrid = []
pasi_hibrid_entropy = []
pasi_gauss_seidel_entropy = []
pasi_jacobi_parallel = []
pasi_gauss_seidel_analytic = []
pasi_sor_analytic = []
pasi_dp_0 = []
pasi_dp_1 = []
pasi_hybrid_dp_0 = []
pasi_hybrid_dp_1 = []

err_sol_hibrid = []
err_sol_hibrid_entropy = []
err_sol_gauss_seidel_entropy = []
err_sol_jacobi_parallel = []
err_sol_gauss_seidel_analytic = []
err_sol_sor_analytic = []
err_sol_dp_0 = []
err_sol_dp_1 = []
err_sol_hybrid_dp_0 = []
err_sol_hybrid_dp_1 = []

err_rel_hibrid = []
err_rel_hibrid_entropy = []
err_rel_gauss_seidel_entropy = []
err_rel_jacobi_parallel = []
err_rel_gauss_seidel_analytic = []
err_rel_sor_analytic = []
err_rel_dp_0 = []
err_rel_dp_1 = []
err_rel_hybrid_dp_0 = []
err_rel_hybrid_dp_1 = []

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
        elif w[3] == "solve_sor_analytic_0.844535:":
            pasi_sor_analytic.append(int(w[0]))
        elif w[3] == "gauss_seidel_entropy_dp_0:":
            pasi_dp_0.append(int(w[0]))
        elif w[3] == "gauss_seidel_entropy_dp_1:":
            pasi_dp_1.append(int(w[0]))
        elif w[3] == "hybrid_entropy_dp_0:":
            pasi_hybrid_dp_0.append(int(w[0]))
        elif w[3] == "hybrid_entropy_dp_1:":
            pasi_hybrid_dp_1.append(int(w[0]))

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

    if len(w) >= 5 and w[3] == "solve_sor_analytic_0.844535:":
        if w[2] == "err_fata_de_solutie":
            err_sol_sor_analytic.append(float(w[4]))
        if w[2] == "err_fata_de_ult_val":
            err_rel_sor_analytic.append(float(w[4]))

    if len(w) >= 5 and w[3] == "gauss_seidel_entropy_dp_0:":
        if w[2] == "err_fata_de_solutie":
            err_sol_dp_0.append(float(w[4]))
        if w[2] == "err_fata_de_ult_val":
            err_rel_dp_0.append(float(w[4]))

    if len(w) >= 5 and w[3] == "gauss_seidel_entropy_dp_1:":
        if w[2] == "err_fata_de_solutie":
            err_sol_dp_1.append(float(w[4]))
        if w[2] == "err_fata_de_ult_val":
            err_rel_dp_1.append(float(w[4]))

    if len(w) >= 5 and w[3] == "hybrid_entropy_dp_0:":
        if w[2] == "err_fata_de_solutie":
            err_sol_hybrid_dp_0.append(float(w[4]))
        if w[2] == "err_fata_de_ult_val":
            err_rel_hybrid_dp_0.append(float(w[4]))

    if len(w) >= 5 and w[3] == "hybrid_entropy_dp_1:":
        if w[2] == "err_fata_de_solutie":
            err_sol_hybrid_dp_1.append(float(w[4]))
        if w[2] == "err_fata_de_ult_val":
            err_rel_hybrid_dp_1.append(float(w[4]))
    
print("pasi_hibrid = ", pasi_hibrid, ";")
print("pasi_hibrid_entropy = ", pasi_hibrid_entropy, ";")
print("pasi_gauss_seidel_entropy = ", pasi_gauss_seidel_entropy, ";")
print("pasi_jacobi_parallel = ", pasi_jacobi_parallel, ";")
print("pasi_gauss_seidel_analytic = ", pasi_gauss_seidel_analytic, ";")
print("pasi_sor_analytic = ", pasi_sor_analytic, ";")
print("pasi_dp_0 = ", pasi_dp_0, ";")
print("pasi_dp_1 = ", pasi_dp_1, ";")
print("pasi_hybrid_dp_0 = ", pasi_hybrid_dp_0, ";")
print("pasi_hybrid_dp_1 = ", pasi_hybrid_dp_1, ";")

print("err_sol_hibrid = ", err_sol_hibrid, ";")
print("err_sol_hibrid_entropy = ", err_sol_hibrid_entropy, ";")
print("err_sol_gauss_seidel_entropy = ", err_sol_gauss_seidel_entropy, ";")
print("err_sol_jacobi_parallel = ", err_sol_jacobi_parallel, ";")
print("err_sol_gauss_seidel_analytic = ", err_sol_gauss_seidel_analytic, ";")
print("err_sol_sor_analytic = ", err_sol_sor_analytic, ";")
print("err_sol_dp_0 = ", err_sol_dp_0, ";")
print("err_sol_dp_1 = ", err_sol_dp_1, ";")
print("err_sol_hybrid_dp_0 = ", err_sol_hybrid_dp_0, ";")
print("err_sol_hybrid_dp_1 = ", err_sol_hybrid_dp_1, ";")

print("err_rel_hibrid = ", err_rel_hibrid, ";")
print("err_rel_hibrid_entropy = ", err_rel_hibrid_entropy, ";")
print("err_rel_gauss_seidel_entropy = ", err_rel_gauss_seidel_entropy, ";")
print("err_rel_jacobi_parallel = ", err_rel_jacobi_parallel, ";")
print("err_rel_gauss_seidel_analytic = ", err_rel_gauss_seidel_analytic, ";")
print("err_rel_sor_analytic = ", err_rel_sor_analytic, ";")
print("err_rel_dp_0 = ", err_rel_dp_0, ";")
print("err_rel_dp_1 = ", err_rel_dp_1, ";")
print("err_rel_hybrid_dp_0 = ", err_rel_hybrid_dp_0, ";")
print("err_rel_hybrid_dp_1 = ", err_rel_hybrid_dp_1, ";")

fin.close()
