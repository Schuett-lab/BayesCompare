from BayesCompare import meas_dist_parallel as mdp
import BayesCompare
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import h5py

## Compute the distance

input_dir = "/home/sezan/Documents/BayesCompare/covs_1000_normalized.npy"
# input_dir = '/home/sezan/Documents/BayesCompare/covs_1000.npy'
# input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_resnet50_densesampled_normalized.npy'

output_dir = "/home/sezan/Documents/BayesCompare/parallel_tests_outs"

mdp.measure_dist_parallel(input_dir, output_dir, meas_name="JSD", num_workers=20)

## Check the calculated distance matrix

filename = "/home/sezan/Documents/BayesCompare/parallel_tests_outs/dist_covs_1000_normalized_JSD.hdf5"

with h5py.File(filename, "r") as f:

    f.visititems(lambda name, obj: print(f"  {name}: {type(obj)}"))

    dset = f["dist"]
    data = dset[...]

for i in range(len(data)):
    data[i, i] = 0.0

plt.figure()
plt.imshow(data, "bone", vmin=0, vmax=np.max(data))
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.show()


## Compute distance with the non-parallized method

input_dir = "/home/sezan/Documents/BayesCompare/covs_1000_normalized.npy"

covs = np.load(input_dir)


N = len(covs)

dist = np.zeros((len(covs), len(covs)))

progress_bar = tqdm.tqdm(total=int((N * (N - 1)) / 2))

for i, ci in enumerate(covs):

    for j, cj in enumerate(covs):

        if j > i:

            dist[i, j] = BayesCompare.distances.jsd(ci, cj)

            dist[j, i] = dist[i, j]

            progress_bar.update(1)

plt.figure()
plt.imshow(dist, "bone", vmin=0, vmax=np.max(dist))
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.show()
