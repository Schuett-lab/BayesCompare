from multiprocessing import Process, Queue, JoinableQueue
import h5py
import multiprocessing as mp
import h5py
import numpy  as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import time
import os
import torch
from functools import partial
from itertools import repeat
from multiprocessing import shared_memory
import tqdm

def writer(que, file_dir, total_num_ops):
    
    progress_bar = tqdm.tqdm(total=int(total_num_ops))
    
    with h5py.File(file_dir, 'r+') as f:
        
        res_dset = f["dist"]
        
        while 1:
                    
            item = que.get()
            
            if item is None:
                break
            
            res_dset[item[0][0], item[0][1]] = item[1]
            res_dset[item[0][1], item[0][0]] = item[1]
            
            f.flush()
            
            progress_bar.update(1)

## corresponds to computation
def wasserstein(sigma1, sigma2, mu1=None, mu2=None):
    
    #these conditions do not check for one mean is non zero and other is zero !!!
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

def select_measure(meas_name):
    
    if meas_name=='wasserstein':
        measure = wasserstein
    
    return measure

def worker(input_queue, output_queue, shared_mem_name, covs_shape, covs_dtype, meas_name):
    
    created_sh_mem = shared_memory.SharedMemory(name=shared_mem_name)
    covs = np.ndarray(covs_shape, dtype=covs_dtype, buffer=created_sh_mem.buf)
    
    measure = select_measure(meas_name)
    
    while True:
        item = input_queue.get()
        
        if item is None: # Sentinel to signal the end of processing
            break
        else:
            idx = item
            
            processed_data = measure(covs[idx[0], :, :], covs[idx[1], :, :])

            output_queue.put((idx, processed_data))
            
            
        #created_sh_mem.close() ## not sure???


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

def starter(input_queue, indices):
    
    for i, j in indices:
            
        idx = (i, j)
        input_queue.put(idx) 
        
        

def create_shared(covs):
    sh_mem = shared_memory.SharedMemory(create=True, size=covs.nbytes)
    covs_shared_np = np.ndarray(covs.shape, dtype=covs.dtype, buffer=sh_mem.buf)
    covs_shared_np[:] = covs[:]
    
    return sh_mem, covs_shared_np


if __name__ == "__main__":
    
    input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_normalized.npy'
    #input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_resnet50_densesampled_normalized.npy'
    #input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_all_resnets_all_layers_normalized.npy'
    output_file_dir = '/home/sezan/Documents/BayesCompare/parallel_tests_outs/shared_mem_deneme4.hdf5'
    
    covs = np.array(np.load(input_dir))
    
    shared_mem, shared_covs = create_shared(covs)
    
    meas_name = 'wasserstein'
    
    N=len(covs)
    
    num_processes= int((N*(N-1))/2)
    
    num_workers = mp.cpu_count()-1
    
    with mp.Manager() as manager:
        
        input_queue = manager.Queue(2 * num_workers)
        output_queue = manager.Queue(2 * num_workers)
        
        indices = check_saved_hdf(output_file_dir, N)
        
        p_starter = Process(target=starter, args=(input_queue, indices))
        
        p_starter.start()
        workers = []
        
        start = time.time()
        
        for _ in range(num_workers):
            
            p = Process(target=worker, args=(input_queue, output_queue, shared_mem.name, covs.shape, covs.dtype, meas_name))
            p.start()
            
            workers.append(p)
        
        p_writer = Process(target=writer, args=(output_queue, output_file_dir, num_processes))
        
        p_writer.start()
        p_starter.join()
        
        for _ in range(num_workers):
            input_queue.put(None) # Send sentinel to workers
            
        for p in workers:
            
            p.join()
            
        end = time.time()
        
        shared_mem.close()
        shared_mem.unlink()
        
        output_queue.put((None)) # Send sentinel to writer
        p_writer.join()
    
    
    print(f"Total duration is: {end-start} for {len(indices)} operations - Sharedmem")
    
## Total duration is: 297.09886598587036 for 300 operations - Sharedmem
