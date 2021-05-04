function [] = generic_scatter_gse_vs_hgse ()pasi_hibrid =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41] ;pasi_hibrid_entropy =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] ;pasi_gauss_seidel_entropy =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36] ;pasi_jacobi_parallel =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114] ;pasi_gauss_seidel_analytic =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73] ;err_sol_hibrid =  [39.5902, 18.7107, 9.13877, 5.29361, 3.8079, 2.77787, 2.18679, 1.7815, 1.46722, 1.20609, 0.891739, 0.59362, 0.362305, 0.2068, 0.112593, 0.0591427, 0.0295418, 0.0130158, 0.00816784, 0.00668437, 0.00556046, 0.00461333, 0.00384633, 0.00310802, 0.00230282, 0.00160029, 0.00105842, 0.000671994, 0.000410449, 0.000239786, 0.000131725, 6.55324e-05, 2.68135e-05, 1.55361e-05, 1.30029e-05, 1.08938e-05, 9.11656e-06, 7.6343e-06, 6.02768e-06, 4.44987e-06, 3.12113e-06] ;err_sol_hibrid_entropy =  [39.5902, 18.7107, 9.13871, 4.72168, 3.02455, 2.069, 1.49413, 1.07342, 0.759474, 0.527786, 0.344333, 0.253439, 0.16595, 0.116795, 0.0780833, 0.0538961, 0.037353, 0.0285827, 0.0166843, 0.00972136, 0.00646429, 0.00475177, 0.0032884, 0.00223838, 0.00141733, 0.00127625, 0.000895319, 0.000620958, 0.000448878, 0.000318073, 0.000223953, 0.000149856, 0.000105407, 8.5224e-05, 5.7746e-05, 4.40392e-05, 3.01638e-05, 2.13409e-05, 1.47127e-05, 9.861e-06, 7.32044e-06, 5.35571e-06, 3.93768e-06] ;err_sol_gauss_seidel_entropy =  [39.5902, 18.7107, 8.99039, 3.49385, 1.42881, 0.792706, 0.471612, 0.302733, 0.230156, 0.180326, 0.125856, 0.0746088, 0.0435849, 0.0280393, 0.0182712, 0.0123345, 0.00878754, 0.00731731, 0.00467524, 0.00287093, 0.00187855, 0.00128434, 0.000850704, 0.000541917, 0.000422011, 0.000277024, 0.000168236, 0.000101138, 6.6319e-05, 4.6372e-05, 3.02416e-05, 1.79971e-05, 1.51988e-05, 9.67789e-06, 5.67571e-06, 3.46109e-06] ;err_sol_jacobi_parallel =  [39.5902, 18.7107, 9.13887, 5.2951, 3.81114, 2.7905, 2.20836, 1.80452, 1.49852, 1.25158, 1.02933, 0.895129, 0.787823, 0.681594, 0.601426, 0.517129, 0.450957, 0.395081, 0.346971, 0.303699, 0.261826, 0.226677, 0.202677, 0.182498, 0.161794, 0.142687, 0.127059, 0.112886, 0.100443, 0.0889671, 0.0785971, 0.0694496, 0.0614988, 0.0546096, 0.0486004, 0.0432974, 0.0385666, 0.0343198, 0.0305047, 0.0270888, 0.0240455, 0.0213461, 0.0189577, 0.0168449, 0.0149734, 0.0133119, 0.0118342, 0.0105186, 0.00934725, 0.00830503, 0.00737856, 0.00655561, 0.00582491, 0.00517613, 0.00459991, 0.00408793, 0.00363289, 0.00322839, 0.00286882, 0.00254923, 0.00226522, 0.00201286, 0.00178865, 0.00158943, 0.00141242, 0.00125513, 0.00111535, 0.000991133, 0.000880742, 0.000782643, 0.000695469, 0.000618006, 0.000549173, 0.000488008, 0.000433656, 0.000385357, 0.000342438, 0.000304298, 0.000270406, 0.000240289, 0.000213526, 0.000189744, 0.00016861, 0.000149831, 0.000133143, 0.000118314, 0.000105137, 9.34267e-05, 8.30211e-05, 7.37744e-05, 6.55576e-05, 5.82559e-05, 5.17675e-05, 4.60018e-05, 4.08782e-05, 3.63253e-05, 3.22795e-05, 2.86843e-05, 2.54895e-05, 2.26505e-05, 2.01278e-05, 1.7886e-05, 1.58939e-05, 1.41237e-05, 1.25506e-05, 1.11527e-05, 9.91058e-06, 8.80677e-06, 7.82589e-06, 6.95426e-06, 6.17971e-06, 5.49143e-06, 4.87981e-06, 4.33631e-06] ;err_sol_gauss_seidel_analytic =  [39.5902, 18.7107, 9.13738, 5.28017, 3.78587, 2.72604, 2.12404, 1.71044, 1.39089, 1.09737, 0.808148, 0.683944, 0.596728, 0.493946, 0.411283, 0.343334, 0.286294, 0.231906, 0.167495, 0.144577, 0.124365, 0.102858, 0.0852662, 0.0711013, 0.0591637, 0.0492167, 0.0370772, 0.0294786, 0.0249714, 0.0213269, 0.0176258, 0.0146457, 0.0122093, 0.0101575, 0.00813634, 0.0059588, 0.00510422, 0.00440682, 0.00362938, 0.00301164, 0.0025148, 0.00209233, 0.00173251, 0.00128662, 0.00104624, 0.000885017, 0.000752215, 0.000620473, 0.000517073, 0.000431539, 0.000358839, 0.000284278, 0.000211613, 0.000179835, 0.000155938, 0.000128446, 0.000106342, 8.89414e-05, 7.39132e-05, 6.09141e-05, 4.45745e-05, 3.70868e-05, 3.1512e-05, 2.65024e-05, 2.18953e-05, 1.82934e-05, 1.5233e-05, 1.26838e-05, 9.9185e-06, 7.50487e-06, 6.32776e-06, 5.51072e-06, 4.54339e-06] ;err_rel_hibrid =  [47.4491, 39.5756, 21.2064, 12.7033, 8.33678, 6.35727, 4.83117, 3.87608, 3.18646, 2.64916, 2.09783, 1.48536, 0.955925, 0.569105, 0.319393, 0.171736, 0.0886845, 0.0425576, 0.0178449, 0.0145821, 0.0120675, 0.0100205, 0.00833989, 0.006952, 0.00541084, 0.00390311, 0.00265871, 0.00173042, 0.00108244, 0.000650235, 0.000371511, 0.000197257, 9.23459e-05, 3.38795e-05, 2.81023e-05, 2.36152e-05, 1.97628e-05, 1.6498e-05, 1.3662e-05, 1.04775e-05, 7.57099e-06] ;err_rel_hibrid_entropy =  [47.4491, 39.5756, 21.2062, 11.6487, 7.11617, 4.62354, 3.45701, 2.40523, 1.71346, 1.05857, 0.732246, 0.56215, 0.379671, 0.256089, 0.173429, 0.117449, 0.0785737, 0.0563217, 0.0383791, 0.0233765, 0.0147943, 0.0100443, 0.00734777, 0.00514569, 0.00322238, 0.0026732, 0.00197239, 0.00142118, 0.000985026, 0.000708806, 0.000508782, 0.000346836, 0.000223962, 0.000175347, 0.000125457, 9.53527e-05, 6.79321e-05, 4.5401e-05, 3.35195e-05, 2.29079e-05, 1.48873e-05, 1.17691e-05, 8.80835e-06] ;err_rel_gauss_seidel_entropy =  [47.4491, 39.5756, 21.1673, 9.96457, 4.04006, 1.77826, 1.02436, 0.689544, 0.433813, 0.378916, 0.254982, 0.168704, 0.0968004, 0.0635469, 0.041942, 0.0277498, 0.0195398, 0.0145667, 0.0102849, 0.00633624, 0.00438503, 0.00287797, 0.00192783, 0.00130407, 0.00085517, 0.000634477, 0.000407642, 0.000227005, 0.000151039, 9.99201e-05, 7.13286e-05, 4.6242e-05, 3.04046e-05, 2.48767e-05, 1.3587e-05, 7.38363e-06] ;err_rel_jacobi_parallel =  [47.4491, 39.5756, 21.2068, 12.7059, 8.34338, 6.37584, 4.87146, 3.9299, 3.24611, 2.71605, 2.22846, 1.92446, 1.64647, 1.45494, 1.28049, 1.10637, 0.960001, 0.842287, 0.738681, 0.650409, 0.565525, 0.487258, 0.428995, 0.385174, 0.344292, 0.303776, 0.269275, 0.239762, 0.213286, 0.18941, 0.167564, 0.148047, 0.130948, 0.116108, 0.10321, 0.0918978, 0.081864, 0.0728864, 0.0648245, 0.0575934, 0.0511343, 0.0453916, 0.0403038, 0.0358026, 0.0318183, 0.0282853, 0.0251461, 0.0223528, 0.0198658, 0.0176523, 0.0156836, 0.0139342, 0.0123805, 0.011001, 0.00977604, 0.00868784, 0.00772082, 0.00686128, 0.00609721, 0.00541805, 0.00481445, 0.00427808, 0.00380151, 0.00337808, 0.00300186, 0.00266756, 0.00237048, 0.00210648, 0.00187187, 0.00166338, 0.00147811, 0.00131348, 0.00116718, 0.00103718, 0.000921663, 0.000819013, 0.000727795, 0.000646736, 0.000574704, 0.000510695, 0.000453814, 0.000403269, 0.000358354, 0.000318441, 0.000282974, 0.000251457, 0.000223451, 0.000198563, 0.000176448, 0.000156795, 0.000139332, 0.000123813, 0.000110023, 9.77693e-05, 8.688e-05, 7.72035e-05, 6.86048e-05, 6.09637e-05, 5.41737e-05, 4.814e-05, 4.27783e-05, 3.80137e-05, 3.37799e-05, 3.00175e-05, 2.66743e-05, 2.37034e-05, 2.10633e-05, 1.87173e-05, 1.66327e-05, 1.47802e-05, 1.3134e-05, 1.16711e-05, 1.03712e-05, 9.21612e-06] ;err_rel_gauss_seidel_analytic =  [47.4491, 39.5756, 21.2012, 12.6807, 8.29085, 6.26462, 4.69215, 3.73649, 3.02102, 2.48057, 1.81575, 1.48166, 1.27098, 1.06499, 0.887085, 0.742735, 0.618789, 0.516636, 0.395035, 0.307747, 0.26108, 0.224598, 0.185526, 0.15395, 0.12815, 0.106556, 0.0862939, 0.0621763, 0.053687, 0.0461477, 0.0381673, 0.0316359, 0.026379, 0.0219471, 0.0182595, 0.0137597, 0.0109357, 0.00926403, 0.00791279, 0.00653968, 0.00543398, 0.00452994, 0.00376871, 0.00301913, 0.00221077, 0.00189386, 0.00163503, 0.00134664, 0.00111741, 0.000933054, 0.000776317, 0.000642843, 0.000477469, 0.00038817, 0.00032836, 0.000279101, 0.000230225, 0.000191849, 0.000160115, 0.000133141, 0.000105489, 7.85114e-05, 6.67267e-05, 5.78573e-05, 4.76567e-05, 3.94566e-05, 3.29999e-05, 2.74243e-05, 2.26023e-05, 1.6542e-05, 1.376e-05, 1.16911e-05, 9.83355e-06] ;  n = 100;  limit_for_oy = 9000;    %pasi_jacobi_parallel = pasi_jacobi_parallel(:, 1:n);  %err_sol_jacobi_parallel = err_sol_jacobi_parallel(:, 1:n);    scatter(pasi_jacobi_parallel, log10(min(limit_for_oy, err_sol_jacobi_parallel)));  hold;  scatter(pasi_gauss_seidel_analytic, log10(min(limit_for_oy, err_sol_gauss_seidel_analytic)));  scatter(pasi_hibrid, log10(min(limit_for_oy, err_sol_hibrid)));  scatter(pasi_gauss_seidel_entropy, log10(min(limit_for_oy, err_sol_gauss_seidel_entropy)));  scatter(pasi_hibrid_entropy, log10(min(limit_for_oy, err_sol_hibrid_entropy)));    mlg = legend("location", "northeast");  mlg = legend({"Jacobi Parallel", "Gauss Seidel Analytic", "Hybrid", "Gauss Seidel Entropy", "Hybrid entropy"});  set (mlg, "fontsize", 10);  xlabel ("Number of Iterations");  ylabel ("Relative error to wanted solution (Infinite norm, logaritmated)");    set (gca, "xminorgrid", "on");  set (gca, "yminorgrid", "on");  set(gca, 'xtick', -1000:5:1000);  set(gca, 'ytick', -1000:1:1000);  title ("All methods have bucketsize coefficient sqrt(n)/2, A X 1000, b X 1000");endfunction