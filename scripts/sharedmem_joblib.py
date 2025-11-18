import numpy as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import tqdm
import os
import glob
import h5py
from joblib import Parallel, delayed, Memory
import torch
from multiprocessing import shared_memory
import time
import multiprocessing as mp
from multiprocessing import Queue

def wasserstein(sigma1, sigma2, mu1=None, mu2=None):
    
    # these conditions do not check for one mean is non zero and other is zero !!!
    if mu1 is not None and mu2 is not None:
        means_term = np.linalg.norm(mu1 - mu2, 2)**2
    else:
        means_term=0
        
    sig1_sqrt = scipy.linalg.sqrtm(sigma1)
    sig1_sig2_sqrt = scipy.linalg.sqrtm(sig1_sqrt @ sigma2 @ sig1_sqrt)
    tr_term = sigma1 + sigma2 - 2*(sig1_sig2_sqrt) 
    d_sq = means_term + np.trace(tr_term)
    
    if d_sq<0 and d_sq>-1e-7:
        d_sq = 0
    
    elif d_sq < -1e-7:
        raise ValueError(f"Wasserstein distance cannot be negative. Value is: {d_sq}")
    
    return d_sq**0.5


def trace_norm(sigma, eye_w=0.001):
    
    if eye_w == 0:
        A = sigma
    
    else:
        A = ((1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)) + (eye_w * np.eye(sigma.shape[0]))
    
    return A

def chunk_pairs(pairs, chunksize):
    
    chunks = []
    
    for k in range(int(np.ceil(len(pairs)/chunksize))):
        chunks.append(pairs[k*chunksize:(k+1)*chunksize])
        
    return chunks

def select_measure(meas_name):
    
    if meas_name=='wasserstein':
        measure = wasserstein
    
    return measure


def check_saved_hdf(hdf_dir, N):
    
    if os.path.exists(hdf_dir):
        
        with h5py.File(hdf_dir, 'r') as f:
            
            dist = f["dist"][...]

            tril_idx = np.tril_indices(dist.shape[0], k=-1)
            nan_mask = np.isnan(dist[tril_idx])
            indices = [[int(i), int(j)] for i, j in zip(tril_idx[0][nan_mask], tril_idx[1][nan_mask])]

    else:
        
        with h5py.File(hdf_dir, 'w') as f:
            
            init_mtx =  np.empty((N,N)) * np.nan
                
            dist_dset = f.create_dataset("dist", shape=(N,N), data=init_mtx)
            indices = np.array([(i, j) for j in range(N) for i in range(j + 1, N)])
            
            f.flush()
         
    return indices


def writer(file_dir, que):
    
    with h5py.File(file_dir, 'r+') as f:
        
        res_dset = f["dist"]
        
        while 1:
            
            item = que.get()
            
            #print(item)
            
            if item is None:
                break
            
            res_dset[item[0], item[1]] = item[2]
            res_dset[item[1], item[0]] = item[2]
            
            f.flush()

def parallel_measure_dis_joblib(covs, measure_name, num_workers, b=1/100): # what should be n-jobs default?
    
    N=len(covs)
    
    output_path = '/home/sezan/Documents/BayesCompare/parallel_tests_outs/shared_memjoblib_medium_1.hdf5'
    
    indices = check_saved_hdf(output_path, N)

    measure = select_measure(measure_name)
    
    manager = mp.Manager()
    output_queue = manager.Queue(2 * num_workers)
    
    writer_procc = mp.Process(target=writer, args=(output_path, output_queue))
    writer_procc.start()
        
    def compute_pairwise_dist(i, j):
        val = measure(covs[i], covs[j])
        # print(isinstance(covs, np.memmap))
        # print(type(output_queue))
        output_queue.put((i, j, val))
        #return (i, j, val)
    
    #upper_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    #dist = np.zeros((N, N), dtype=float)
    
    start = time.time()
        
    Parallel(n_jobs=num_workers, backend='loky', verbose=10)(delayed(compute_pairwise_dist)(i, j) for i, j in indices)

    # for i, j, v in dist_list:
    #     dist[i, j] = v
    #     dist[j, i] = v
        
    # np.save(output_path, dist)
    
    end = time.time()
    
    output_queue.put(None)
    writer_procc.join()
    
    print(f"Total duration is: {end-start} for {int((N*(N-1))/2)} operations - joblib")
    
    #return dist


if __name__ == "__main__":
    
    #input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_normalized.npy'
    input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_resnet50_densesampled_normalized.npy'
    #input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_all_resnets_all_layers_normalized.npy'
    
    covs = np.load(input_dir)
    
    measure_name = 'wasserstein'
    
    dists = parallel_measure_dis_joblib(covs, measure_name, num_workers=20)

# old one: Total duration is: 180.6669204235077 for 300 operations - joblib


## small covs list
# [Parallel(n_jobs=20)]: Done 300 out of 300 | elapsed:  1.0min finished
# Total duration is: 60.70717477798462 for 300 operations - joblib