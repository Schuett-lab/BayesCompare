import multiprocessing as mp
import h5py
import numpy  as np
import scipy.linalg
import time
import tqdm
from joblib import Parallel, delayed

output_file_dir='/home/sezan/Documents/BayesCompare/dists.hdf5'

input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_all_resnets_all_layers.npy'

def init_vars(mtx_list, out_q):
    global G_Q, G_MTX
    G_Q = out_q
    G_MTX = mtx_list
    
def trace_norm(sigma, eye_w=0.001):
    
    if eye_w == 0:
        A = sigma
    
    else:
        A = ((1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)) + (eye_w * np.eye(sigma.shape[0]))
    
    return A

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

## Parlallel Computation
'''
def dist_workers(ind_tuple):
    
    i = ind_tuple[0]
    j = ind_tuple[1]
    mi = G_MTX[i]
    mj = G_MTX[j]
    
    res = wasserstein(trace_norm(mi, 10/11), trace_norm(mj, 10/11))
    
    G_Q.put((i, j, res))
    

def writer(file_dir, que, max_dim):
    
    with h5py.File(file_dir, 'w') as f:
        
        dset = f.create_dataset("results", shape=(max_dim,), dtype=np.dtype([('i', 'i4'), ('j', 'i4'), ('res', 'f8')]))
        row=0
        
        while 1:
            
            m = que.get()
            
            if m is None:
                break
            
            dset[row] = m
            f.flush()
            row += 1


if __name__ == "__main__":
    
    mtx_list = np.load(input_dir)
    
    N = len(mtx_list)
    
    manager = mp.Manager()
    queues = manager.Queue()
    
    max_dim = int((N*(N-1))/2)
    
    writer_procc = mp.Process(target=writer, args=(output_file_dir, queues, max_dim), daemon=True)
    writer_procc.start()
    
    indices = [(i, j) for j in range(N) for i in range(j + 1, N)]
    
    start = time.time()
    
    with mp.Pool(processes=mp.cpu_count()-1, initializer=init_vars, initargs=(mtx_list, queues)) as pool:

        pool.map_async(dist_workers, indices)
            
        pool.close()
        pool.join()
    
    end = time.time()
    
    queues.put(None)
    writer_procc.join()

    print(f"Total duration is: {end-start} for {max_dim} operations - map_async")
'''
## Result: Total duration is: 93.60919213294983 for 300 operations - map_async
    
# Normal Computation

input_dir = '/home/sezan/Documents/BayesCompare/covs_1000.npy'

covs = np.load(input_dir)


alpha = 10/11

dist = np.zeros((len(covs), len(covs)))

#start = time.time()

for i, ci in enumerate(covs):
    
    sig1 = trace_norm(ci, eye_w=alpha)
    
    for j, cj in enumerate(covs):
        
        if j > i:
            
            sig2 = trace_norm(cj, eye_w=alpha)
            
            start = time.time()
            wasserstein(sig1, sig2)
            end = time.time()
            
            print(f"Each distance calculation duration is: {end-start}")

#end = time.time()

#print(f"Total duration is: {end-start} normal")

## Result: Total duration is: 186.99659395217896 normal


# Normal Computation with trace normalized matrices
'''
output_file_dir='/home/sezan/Documents/BayesCompare/dists.hdf5'

input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_all_resnets_all_layers.npy'

def init_vars(mtx_list, out_q):
    global G_Q, G_MTX
    G_Q = out_q
    G_MTX = mtx_list
    
def trace_norm(sigma, eye_w=0.001):
    
    if eye_w == 0:
        A = sigma
    
    else:
        A = ((1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)) + (eye_w * np.eye(sigma.shape[0]))
    
    return A

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
    

def dist_workers(ind_tuple):
    
    i = ind_tuple[0]
    j = ind_tuple[1]
    mi = G_MTX[i]
    mj = G_MTX[j]
    
    res = wasserstein(mi, mj)
    
    G_Q.put((i, j, res))
    

def writer(file_dir, que, max_dim):
    
    with h5py.File(file_dir, 'w') as f:
        
        dset = f.create_dataset("results", shape=(max_dim,), dtype=np.dtype([('i', 'i4'), ('j', 'i4'), ('res', 'f8')]))
        row=0
        
        while 1:
            
            m = que.get()
            
            if m is None:
                break
            
            dset[row] = m
            f.flush()
            row += 1


if __name__ == "__main__":
    
    mtx_list = np.load(input_dir)
    
    N = len(mtx_list)
    
    norm_sigmas = Parallel(n_jobs=mp.cpu_count()-1, backend="loky")(delayed(trace_norm)(mtx_list[i], 10/11) for i in range(N))
    
    manager = mp.Manager()
    queues = manager.Queue()
    
    max_dim = int((N*(N-1))/2)
    
    writer_procc = mp.Process(target=writer, args=(output_file_dir, queues, max_dim), daemon=True)
    writer_procc.start()
    
    indices = [(i, j) for j in range(N) for i in range(j + 1, N)]
    
    start = time.time()
    
    with mp.Pool(processes=mp.cpu_count()-1, initializer=init_vars, initargs=(norm_sigmas, queues)) as pool:

        pool.map_async(dist_workers, indices)
            
        pool.close()
        pool.join()
    
    end = time.time()
    
    queues.put(None)
    writer_procc.join()

    print(f"Total duration is: {end-start} for {max_dim} operations - map_async with precalculated normalized mtx")'''

## Result: Total duration is: 94.5107831954956 for 300 operations - map_async with precalculated normalized mtx