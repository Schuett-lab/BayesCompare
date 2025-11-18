from BayesCompare import meas_dist_parallel as mdp

## Compute the distance

input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_normalized.npy'
    
output_dir = '/home/sezan/Documents/BayesCompare/parallel_tests_outs'

mdp.measure_dist_parallel(input_dir, output_dir, meas_name='wasserstein', num_workers=20)

## Check the calculated distance matrix

import matplotlib.pyplot as plt
import numpy as np
import h5py

filename = "/home/sezan/Documents/BayesCompare/parallel_tests_outs/dist_covs_1000_normalized_wasserstein.hdf5"

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
1+1