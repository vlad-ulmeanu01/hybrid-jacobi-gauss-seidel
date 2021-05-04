function [] = generic_scatter_sol ()
pasi_hibrid =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43] ;
pasi_hibrid_entropy =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100] ;
err_sol_hibrid =  [159.075, 73.3255, 28.7241, 17.4878, 12.9517, 9.86302, 7.96577, 6.41281, 5.2614, 4.3216, 3.21417, 2.16595, 1.34361, 0.782536, 0.438216, 0.240702, 0.130019, 0.0669385, 0.0299568, 0.0186706, 0.015337, 0.0125872, 0.0103725, 0.00860265, 0.00675033, 0.0049183, 0.00339357, 0.00225228, 0.00145035, 0.000907777, 0.000549188, 0.000316943, 0.000170412, 8.15482e-05, 3.07382e-05, 2.08246e-05, 1.73332e-05, 1.44554e-05, 1.20248e-05, 9.9724e-06, 7.72167e-06, 7.84095e-06, 7.83219e-06] ;
err_sol_hibrid_entropy =  [159.075, 73.3255, 28.7256, 17.3401, 12.1384, 8.86027, 6.50808, 4.56703, 3.22256, 2.42805, 1.76361, 1.03523, 0.708889, 0.502685, 0.34405, 0.232468, 0.18273, 0.12492, 0.129257, 0.0532355, 0.185202, 0.0802015, 0.0160077, 0.0131341, 1.59086, 0.213665, 0.0354975, 0.0154443, 0.0124691, 0.00678222, 0.0628998, 0.0319749, 0.33388, 0.026606, 0.0080699, 0.00451638, 0.010812, 0.0231094, 0.0313545, 0.00652287, 0.179288, 0.0386187, 0.05401, 0.108178, 0.665492, 0.0595568, 0.484211, 0.0263593, 0.00716493, 0.0154652, 0.0453549, 0.00236268, 0.00861213, 0.0226169, 0.0350783, 0.16802, 0.318416, 0.0380867, 0.174331, 0.0298728, 0.0186684, 0.00875716, 0.037826, 0.338052, 0.0315158, 2.10269, 0.369306, 0.195514, 0.0404227, 0.0280653, 0.0227942, 0.2625, 0.0105717, 0.016101, 0.637472, 0.337529, 0.0559875, 0.0268839, 0.016849, 0.0366193, 1.95898, 0.146713, 0.0341761, 0.0158554, 0.00886951, 0.0877304, 0.0111732, 0.167766, 0.46269, 0.402505, 0.0633304, 0.0197368, 0.0592968, 0.163529, 0.213918, 0.00729056, 0.00371739, 0.0216357, 0.0531529, 0.0501845] ;
err_rel_hibrid =  [159.215, 159.037, 86.2105, 39.8512, 29.681, 21.8557, 17.297, 13.9443, 11.4808, 9.49521, 7.52341, 5.37823, 3.50955, 2.12614, 1.22075, 0.678918, 0.370722, 0.196958, 0.0968953, 0.0405201, 0.0335876, 0.0274471, 0.0225939, 0.0186692, 0.0153475, 0.0116686, 0.00831188, 0.00564528, 0.00370253, 0.00235813, 0.00145696, 0.000866131, 0.000487354, 0.00025196, 0.000112286, 4.47375e-05, 3.74616e-05, 3.1281e-05, 2.61357e-05, 2.18587e-05, 1.76785e-05, 1.33811e-05, 9.63791e-06] ;
err_rel_hibrid_entropy =  [159.215, 159.037, 86.2114, 39.817, 29.416, 19.2988, 14.7399, 9.90736, 7.16246, 4.66676, 3.23974, 2.50688, 1.64322, 1.10257, 0.746439, 0.505804, 0.344461, 0.250188, 0.191782, 0.169058, 0.13833, 0.214922, 0.0803371, 0.0263284, 1.58974, 1.6394, 0.217791, 0.0410815, 0.0232694, 0.0147261, 0.0626532, 0.0648513, 0.336603, 0.333595, 0.028048, 0.00738684, 0.0126758, 0.0235976, 0.0312073, 0.0311063, 0.178982, 0.179389, 0.0533938, 0.107587, 0.668874, 0.66539, 0.484674, 0.488687, 0.0236264, 0.0145101, 0.0446753, 0.0453296, 0.00863661, 0.0226312, 0.0350356, 0.16804, 0.317787, 0.317713, 0.172217, 0.191521, 0.0400145, 0.0193355, 0.0378927, 0.337984, 0.337035, 2.1102, 2.21801, 0.435363, 0.197418, 0.0642076, 0.043859, 0.254648, 0.256277, 0.0171117, 0.641745, 0.781961, 0.337501, 0.0665193, 0.0395846, 0.039466, 1.96002, 2.02036, 0.165335, 0.0428829, 0.0221251, 0.0876482, 0.0914547, 0.167077, 0.463754, 0.433879, 0.429734, 0.0799873, 0.0701339, 0.16478, 0.214423, 0.214605, 0.00831596, 0.0217959, 0.0533326, 0.0539751] ;

  n = 50;
  limit_for_oy = 9000;
  
  scatter(pasi_hibrid, log10(min(limit_for_oy, err_sol_hibrid)));
  hold;
  scatter(pasi_hibrid_entropy(:, 1:n), log10(min(limit_for_oy, err_sol_hibrid_entropy)(:, 1:n)));
  
  mlg = legend({"hibrid", "hibrid entropy"});
  set (mlg, "fontsize", 10);
  xlabel ("Number of Iterations");
  ylabel ("Relative error to wanted solution (Infinite norm, logaritmated)");
  
  set (gca, "xminorgrid", "on");
  set (gca, "yminorgrid", "on");
  set(gca, 'xtick', -1000:1:1000);
  set(gca, 'ytick', -1000:1:1000);
  title ("Both hybrid and hybrid entropy have bucketsize coefficient sqrt(n)/2");
endfunction