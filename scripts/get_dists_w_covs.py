import numpy as np
from BayesCompare import measure_dist
import matplotlib.pyplot as plt

'''

covs1 = np.load("/home/sezan/Documents/BayesCompare/covs_1000.npy")

covs2 = np.load("/home/sezan/Documents/BayesCompare/covs_1000_resnet50_trial.npy")

covs3 = np.load("/home/sezan/Documents/BayesCompare/covs_example.npy")

save_path = "/home/sezan/Documents/BayesCompare/outputs/dists"

alpha = 10/11

dist_covs1_ws = measure_dist(covs1, meas_name='wasserstein', alpha=alpha)
dist_covs1_hl = measure_dist(covs1, meas_name='hellinger', alpha=alpha)
dist_covs1_mt = measure_dist(covs1, meas_name='matusita', alpha=alpha)
dist_covs1_TVD = measure_dist(covs1, meas_name='TVD', alpha=alpha)
dist_covs1_JSD = measure_dist(covs1, meas_name='JSD', alpha=alpha)
dist_covs1_KL = measure_dist(covs1, meas_name='KL_div', alpha=alpha)
dist_covs1_bh = measure_dist(covs1, meas_name='bhattacharyya', alpha=alpha)

dist_covs2_ws = measure_dist(covs2, meas_name='wasserstein', alpha=alpha)
dist_covs2_hl = measure_dist(covs2, meas_name='hellinger', alpha=alpha)
dist_covs2_mt = measure_dist(covs2, meas_name='matusita', alpha=alpha)
dist_covs2_TVD = measure_dist(covs2, meas_name='TVD', alpha=alpha)
dist_covs2_JSD = measure_dist(covs2, meas_name='JSD', alpha=alpha)
dist_covs2_KL = measure_dist(covs2, meas_name='KL_div', alpha=alpha)
dist_covs2_bh = measure_dist(covs2, meas_name='bhattacharyya', alpha=alpha)

alpha = 2/3

dist_covs3_ws = measure_dist(covs3, meas_name='wasserstein', alpha=alpha)
dist_covs3_hl = measure_dist(covs3, meas_name='hellinger', alpha=alpha)
dist_covs3_mt = measure_dist(covs3, meas_name='matusita', alpha=alpha)
dist_covs3_TVD = measure_dist(covs3, meas_name='TVD', alpha=alpha)
dist_covs3_JSD = measure_dist(covs3, meas_name='JSD', alpha=alpha)
dist_covs3_KL = measure_dist(covs3, meas_name='KL_div', alpha=alpha)
dist_covs3_bh = measure_dist(covs3, meas_name='bhattacharyya', alpha=alpha)

for name, value in list(locals().items()):
    if name.startswith('dist_'):
        np.save(f"{save_path}/{name}.npy", value)
'''

## To confirm that the newly written distances.py gives the same results as in the paper

## Comparison with the examples.py output

'''
# compute directly from covs_example.npy
#covs3 = np.load("/home/sezan/Documents/BayesCompare/covs_example.npy")
#alpha = 2/3

#my_JSD_output = measure_dist(covs3, meas_name='JSD', alpha=alpha)

# or load from the saved output
my_JSD_output = np.load("/home/sezan/Documents/BayesCompare/outputs/dists/dist_covs3_JSD.npy")

heikos_output = np.load("/home/sezan/Documents/BayesCompare/dist_example.npy")

order = [0, 26, 1, 2, 3, 4, 5, 6, 7, 8, 9,
         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 27, 28, 29, 30,
         31, 32, 33, 34, 35, 36, 37, 38]
plt.figure()
plt.imshow(np.sqrt(my_JSD_output[order][:, order]), "bone", vmin=0, vmax=1)
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/example_replication.svg")
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/example_replication.pdf")

plt.figure()
plt.imshow(np.sqrt(heikos_output[order][:, order]), "bone", vmin=0, vmax=1)
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/Heikos_example.svg")
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/Heikos_example.pdf")
'''

## Comparison with Nimages.py output
## This is not correct since in Nimages, 100 images are used not the covs_1000.npy
## So results are not comparable.

'''
my_JSD_output = np.load("/home/sezan/Documents/BayesCompare/outputs/dists/dist_covs1_JSD.npy")

my_TVD_output = np.load("/home/sezan/Documents/BayesCompare/outputs/dists/dist_covs1_TVD.npy")

heikos_output = dist = np.load("/home/sezan/Documents/BayesCompare/dist_rep_all.npy")

plt.figure()
plt.imshow(np.mean(heikos_output[0, 2], 0), cmap="bone")
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/Heikos_jsd_nimages.svg")
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/Heikos_jsd_nimages.pdf")

plt.figure()
plt.imshow(np.mean(heikos_output[1, 2], 0), cmap="bone")
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/Heikos_tvd_nimages.svg")
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/Heikos_tvd_nimages.pdf")


plt.figure()
plt.imshow(np.mean(my_JSD_output), "bone", vmin=0, vmax=1)
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/nimages_jsd_replication.svg")
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/nimages_jsd_replication.pdf")

plt.figure()
plt.imshow(np.mean(my_TVD_output), "bone", vmin=0, vmax=1)
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/nimages_tvd_replication.svg")
plt.savefig("/home/sezan/Documents/BayesCompare/outputs/dists/comparison_figures/nimages_tvd_replication.pdf")
'''