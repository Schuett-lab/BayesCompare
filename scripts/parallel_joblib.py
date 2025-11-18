import numpy as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import tqdm
import os
import glob
from joblib import Parallel, delayed, Memory
import torch
import time
import multiprocessing as mp

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

def parallel_measure_dis_joblib(covs, b=1/100): # what should be n-jobs default?
    
    N=len(covs)
    
    output_path = '/home/sezan/Documents/BayesCompare/parallel_tests_outs/joblib_v3.npy'
    
    #norm_sigmas = Parallel(n_jobs=20, backend="loky")(delayed(trace_norm)(covs[i], alpha) for i in range(N))
        
    def compute_pairwise_dist(mi, mj, meas, i, j):
        val = meas(mi, mj)
        return (i, j, val)
    
    upper_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    per_checkpoint_chunk = chunk_pairs(upper_pairs, int(len(upper_pairs)/300))
    
    dist = np.zeros((N, N), dtype=float)
    
    print(len(per_checkpoint_chunk))
    
    start = time.time()

    for chunk in per_checkpoint_chunk:
        
        dist_list = Parallel(n_jobs=20, backend="loky", verbose=10)(delayed(compute_pairwise_dist)(covs[i], covs[j], wasserstein, i, j) for (i, j) in chunk)
    
        for i, j, v in dist_list:
            dist[i, j] = v
            dist[j, i] = v
            
        np.save(output_path, dist)
        
        dist_list = []
    
    end = time.time()
    
    print(f"Total duration is: {end-start} for {int((N*(N-1))/2)} operations - joblib")
    
    return dist


if __name__ == "__main__":
    
    input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_normalized.npy'
    #input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_resnet50_densesampled_normalized.npy'
    #input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_all_resnets_all_layers_normalized.npy'
    
    covs = np.load(input_dir)
    
    dists = parallel_measure_dis_joblib(covs)

# Total duration is: 6767.381769418716 for 37128 operations - joblib
## [Parallel(n_jobs=20)]: Done 300 out of 300 | elapsed:   50.1s finished