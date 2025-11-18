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
import tqdm 

## Metrics

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


def hellinger(sigma1, sigma2, mu1=None, mu2=None):
    
    d_B = bhattacharyya(sigma1, sigma2, mu1, mu2)
    
    d = np.sqrt(2*(1 - np.exp(-d_B)))
    
    return d

def mahalanobis(sigma1, sigma2, mu1=None, mu2=None): # not the true mahalanobis definition
    
    # these conditions do not check for one mean is non zero and other is zero !!!
    if mu1 is not None and mu2 is not None:
        delta_mu = np.subtract(mu1, mu2)
        d = np.inner(delta_mu, np.matmul((np.linalg.inv(sigma1+sigma2)), delta_mu)) 
    else: 
        d = np.array(0)
    
    return d
    

gen = np.random.Generator(np.random.SFC64(42))

gen_torch = torch.Generator(device='cpu').manual_seed(42)

def jsd(sigma1, sigma2, mu1=None, mu2=None, N=10000, gen=gen):
    
    if mu1 is None and mu2 is None:
        k = sigma1.shape[0]
        A1 = np.linalg.cholesky(sigma1)
        A2 = np.linalg.cholesky(sigma2)
        logdet1 = np.sum(np.log(np.diag(A1)))
        logdet2 = np.sum(np.log(np.diag(A2)))
        # generate random samples from each distribution
        x10 = gen.standard_normal(size=(k, N))
        x1 = mm(1, A1, x10, lower=1)
        x20 = gen.standard_normal(size=(k, N))
        x2 = mm(1, A2, x20, lower=1)
        # compute densities for each
        p1 = - np.sum(x10 ** 2, 0) / 2 - logdet1
        delta21 = scipy.linalg.solve_triangular(A1, x2, lower=True)
        p2 = - np.sum(delta21 ** 2, 0) / 2 - logdet1
        delta12 = scipy.linalg.solve_triangular(A2, x1, lower=True)
        q1 = - np.sum(delta12 ** 2, 0) / 2 - logdet2
        q2 = - np.sum(x20 ** 2, 0) / 2 - logdet2

        # log (P) - log (P + Q)
        term1 = p1 - np.logaddexp(p1, q1)
        term2 = q2 - np.logaddexp(p2, q2)

        jsd = 1 + (np.mean(term1) + np.mean(term2)) / 2 / np.log(2)
        
        
    else:
        k = len(mu1)
        A1 = np.linalg.cholesky(sigma1)
        A2 = np.linalg.cholesky(sigma2)
        # Ainv1 = np.linalg.inv(A1)
        # Ainv2 = np.linalg.inv(A2)
        Ainv1 = scipy.linalg.solve_triangular(A1, np.eye(k), lower=True)
        Ainv2 = scipy.linalg.solve_triangular(A2, np.eye(k), lower=True)
        # generate random samples from each distribution
        x1 = np.expand_dims(mu1, 1) + A1 @ gen.standard_normal(size=(k, N))
        x2 = np.expand_dims(mu2, 1) + A2 @ gen.standard_normal(size=(k, N))
        # compute densities for each
        # removed factor 2 from these as it cancels
        logdet1 = np.sum(np.log(np.diag(A1)))
        logdet2 = np.sum(np.log(np.diag(A2)))
        delta11 = Ainv1 @ (x1 - np.expand_dims(mu1, 1))
        p1 = - np.sum(delta11 ** 2, 0) / 2 - logdet1
        delta21 = Ainv1 @ (x2 - np.expand_dims(mu1, 1))
        p2 = - np.sum(delta21 ** 2, 0) / 2 - logdet1
        delta12 = Ainv2 @ (x1 - np.expand_dims(mu2, 1))
        q1 = - np.sum(delta12 ** 2, 0) / 2 - logdet2
        delta22 = Ainv2 @ (x2 - np.expand_dims(mu2, 1))
        q2 = - np.sum(delta22 ** 2, 0) / 2 - logdet2

        # log (P) - log (P + Q)
        term1 = p1 - np.logaddexp(p1, q1)
        term2 = q2 - np.logaddexp(p2, q2)

        jsd = 1 + (np.mean(term1) + np.mean(term2)) / 2 / np.log(2)
        
    return max(0, jsd)

def tvd(sigma1, sigma2, mu1=None, mu2=None, N=10000, gen=gen):

    if mu1 is not None and mu2 is not None:
        k = len(mu1)
        A1 = np.linalg.cholesky(sigma1)
        A2 = np.linalg.cholesky(sigma2)
        Ainv1 = scipy.linalg.solve_triangular(A1, np.eye(k), lower=True)
        Ainv2 = scipy.linalg.solve_triangular(A2, np.eye(k), lower=True)
        # generate random samples from each distribution
        x1 = np.expand_dims(mu1, 1) + A1 @ gen.standard_normal(size=(k, N))
        x2 = np.expand_dims(mu2, 1) + A2 @ gen.standard_normal(size=(k, N))
        # compute densities for each
        # removed factor 2 from these as it cancels
        logdet1 = np.sum(np.log(np.diag(A1)))
        logdet2 = np.sum(np.log(np.diag(A2)))
        delta11 = Ainv1 @ (x1 - np.expand_dims(mu1, 1))
        p1 = - np.sum(delta11 ** 2, 0) / 2 - logdet1
        delta21 = Ainv1 @ (x2 - np.expand_dims(mu1, 1))
        p2 = - np.sum(delta21 ** 2, 0) / 2 - logdet1
        delta12 = Ainv2 @ (x1 - np.expand_dims(mu2, 1))
        q1 = - np.sum(delta12 ** 2, 0) / 2 - logdet2
        delta22 = Ainv2 @ (x2 - np.expand_dims(mu2, 1))
        q2 = - np.sum(delta22 ** 2, 0) / 2 - logdet2
        f1 = np.maximum(1 - np.exp(q1-p1), 0)
        f2 = np.maximum(1 - np.exp(p2-q2), 0)
        tvd = (np.mean(f1) + np.mean(f2)) / 2
        
    else:
        k = sigma1.shape[0]
        A1 = np.linalg.cholesky(sigma1)
        A2 = np.linalg.cholesky(sigma2)
        logdet1 = np.sum(np.log(np.diag(A1)))
        logdet2 = np.sum(np.log(np.diag(A2)))
        # generate random samples from each distribution
        x10 = gen.standard_normal(size=(k, N))
        x1 = mm(1, A1, x10, lower=1)
        x20 = gen.standard_normal(size=(k, N))
        x2 = mm(1, A2, x20, lower=1)
        # compute densities for each
        p1 = - np.sum(x10 ** 2, 0) / 2 - logdet1
        delta21 = scipy.linalg.solve_triangular(A1, x2, lower=True)
        p2 = - np.sum(delta21 ** 2, 0) / 2 - logdet1
        delta12 = scipy.linalg.solve_triangular(A2, x1, lower=True)
        q1 = - np.sum(delta12 ** 2, 0) / 2 - logdet2
        q2 = - np.sum(x20 ** 2, 0) / 2 - logdet2
        f1 = np.maximum(1 - np.exp(q1-p1), 0)
        f2 = np.maximum(1 - np.exp(p2-q2), 0)
        tvd = (np.mean(f1) + np.mean(f2)) / 2
    
    return max(0, tvd)

## Divergences

def KL_div(sigma1, sigma2, mu1=None, mu2=None):
    
    if mu1 is None:
        mu1 = np.zeros(sigma1.shape[0])
    
    if mu2 is None:
        mu2 = np.zeros(sigma2.shape[0])
    
    delta_mu = np.subtract(mu2, mu1)

    inv_s2 = np.linalg.inv(sigma2)
        
    if (delta_mu<1E-50).all(): # only this condition is tested 
        mean_term = 0
        
    else:  # this condition was not tested
        mean_term = np.transpose(delta_mu) @ inv_s2 @ delta_mu
        
    tr_term = np.trace(inv_s2 @ sigma1)
    
    log_term = np.linalg.slogdet(sigma1)[1]-np.linalg.slogdet(sigma2)[1]
    
    d = (1/2) * (mean_term + tr_term - log_term - sigma1.shape[0])
    
    return d

def bhattacharyya(sigma1, sigma2, mu1=None, mu2=None): 
    
    means_term = mahalanobis(sigma1, sigma2, mu1, mu2)
    log_term = np.linalg.slogdet(np.divide(sigma1 + sigma2, 2))[1] - 0.5*(np.linalg.slogdet(sigma1)[1] + np.linalg.slogdet(sigma2)[1]) # these may also require float64 casting
    
    d = 1/8*means_term + 1/2*log_term
    
    return d
    
def trace_norm(sigma, eye_w=0.001):
    
    if eye_w == 0:
        A = sigma
    
    else:
        A = ((1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)) + (eye_w * np.eye(sigma.shape[0]))
    
    return A

## Parallelization workers

## initialize for only queue as global
def init_vars_func(out_q):
    global G_Q
    G_Q = out_q

## initialize for queue, measure, and mtx_list as global
def init_vars(mtx_list, out_q, measure):
    global G_Q, G_MTX, G_MEAS
    G_Q = out_q
    G_MTX = mtx_list
    G_MEAS = measure
    
def select_measure(meas_name):
    
    if meas_name=='wasserstein':
        measure = wasserstein
        
    elif meas_name=='hellinger':
        measure = hellinger
    
    elif meas_name=='TVD':
        measure = tvd
    
    elif meas_name=='JSD':
        measure = jsd
    
    elif meas_name=='KL_div':
        measure = KL_div
        
    elif meas_name=='bhattacharyya':
        measure = bhattacharyya

    return measure

# worker type that takes the queue, measure, and mtx list as a global constant
def dist_workers_map_async_global(ind_tuple):
     
    i = ind_tuple[0]
    j = ind_tuple[1]
    mi = G_MTX[i]
    mj = G_MTX[j]
    
    if G_MEAS in {jsd, tvd}:
        res = G_MEAS(mi, mj, N=5000)
    else:
        res = G_MEAS(mi, mj)
    
    G_Q.put((i, j, res))

# worker type that takes the queue as a global constant
def dist_workers_map_async_global_queue(iterator_element):
     
    cov_i = iterator_element[1][0]
    cov_j = iterator_element[1][1]
    
    measure = iterator_element[2]
    
    meas = select_measure(measure)
    
    if meas in {jsd, tvd}:
        res = meas(cov_i, cov_j, N=5000)
    else:
        res = meas(cov_i, cov_j)

    G_Q.put((int(iterator_element[0][0]), int(iterator_element[0][1]), res))

# worker type that takes the queue and measure as partial function
def dist_workers_map_async_partial(iterator_element, **kwargs):
     
    cov_i = iterator_element[1][0]
    cov_j = iterator_element[1][1]
    
    queue = kwargs["queue"]
    meas = kwargs["measure"]
    
    if meas in {jsd, tvd}:
        res = meas(cov_i, cov_j, N=5000)
    else:
        res = meas(cov_i, cov_j)
    
    queue.put((int(iterator_element[0][0]), int(iterator_element[0][1]), res))


def dist_workers_starmap_async(indices, covs, measure, queue):
     
    cov_i = covs[indices[0]]
    cov_j = covs[indices[1]]
    
    if measure in {jsd, tvd}:
        res = measure(cov_i, cov_j, N=5000)
    else:
        res = measure(cov_i, cov_j)
    
    queue.put((int(indices[0]), int(indices[1]), res))


def writer(file_dir, que, total_num_ops):
    
    progress_bar = tqdm.tqdm(total=int(total_num_ops))
    
    with h5py.File(file_dir, 'r+') as f:
        
        res_dset = f["dist"]
        
        while 1:
            
            item = que.get()
            
            if item is None:
                break
            
            res_dset[item[0], item[1]] = item[2]
            res_dset[item[1], item[0]] = item[2]
            
            f.flush()
            
            progress_bar.update(1)

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


def meas_dist_map_async_global(covs, mean=None, meas_name='TVD', alpha=None, b=1/100):
    
    N=len(covs) # maybe checking if covs is a numpy array of pickle file'
        
    # Global consts
    
    output_file_dir='/home/sezan/Documents/BayesCompare/parallel_tests_outs/parallel_tests_global_allayers.hdf5'

    indices = check_saved_hdf(output_file_dir, N)
    
    nof_operations = len(indices)

    # Global consts
    
    measure = select_measure(meas_name)
        
    if len(indices)==0:
        print("Distance already calculated.")
        
    else: 
        
        manager = mp.Manager()
        queues = manager.Queue()
        
        writer_procc = mp.Process(target=writer, args=(output_file_dir, queues, nof_operations), daemon=True)
        writer_procc.start()
        
        start = time.time()

        with mp.Pool(processes=mp.cpu_count()-1, initializer=init_vars, initargs=(covs, queues, measure)) as pool:

            pool.map_async(dist_workers_map_async_global, iter(indices))
                
            pool.close()
            pool.join()
        
        end = time.time()
        
        queues.put(None)
        writer_procc.join()

        print(f"Total duration is: {end-start} for {len(indices)} operations - map_async + global pass")


def meas_dist_map_async_global_queue(covs, mean=None, meas_name='TVD', alpha=None, b=1/100):
    
    N=len(covs) # maybe checking if covs is a numpy array of pickle file
         
    # Global consts - queue
    
    output_file_dir='/home/sezan/Documents/BayesCompare/parallel_tests_outs/parallel_tests_global_queue_allayers.hdf5'

    indices = check_saved_hdf(output_file_dir, N)
    
    nof_operations = len(indices)
    
    cov_pairs = [(covs[i], covs[j]) for (i, j) in indices]

    # Global consts - queue
    
    meas_name_list = [meas_name]*len(indices)
    
    iterator = zip(indices, cov_pairs, meas_name_list)
    
    if alpha==None:
        alpha = N * b / (1 + (N * b))
        
    if len(indices)==0:
        print("Distance already calculated.")
        
    else: 
        
        worker_count = (mp.cpu_count()-1)
        
        #manager = mp.Manager()
        #queues = manager.Queue(2 * worker_count)
        
        queues = mp.Queue(2 * worker_count)
        
        writer_procc = mp.Process(target=writer, args=(output_file_dir, queues, nof_operations), daemon=True)
        writer_procc.start()
        
        start = time.time()

        with mp.Pool(processes=mp.cpu_count()-1, initializer=init_vars_func, initargs=(queues,)) as pool:

            pool.map_async(dist_workers_map_async_global_queue, iterator, chunksize=200)
                
            pool.close()
            pool.join()
        
        end = time.time()
        
        queues.put(None)
        writer_procc.join()

        print(f"Total duration is: {end-start} for {len(indices)} operations - map_async + global queue pass")
    

def meas_dist_map_async_partial(covs, mean=None, meas_name='TVD', alpha=None, b=1/100):
    
    N=len(covs) # maybe checking if covs is a numpy array of pickle file
    
    output_file_dir='/home/sezan/Documents/BayesCompare/parallel_tests_outs/parallel_tests_partial_allayers.hdf5'    
    
    indices = check_saved_hdf(output_file_dir, N)
    
    nof_operations = len(indices)
    
    cov_pairs = [(covs[i], covs[j]) for (i, j) in indices]
    
    meas = select_measure(meas_name)
    
    iterator = zip(indices, cov_pairs)
    
    if alpha==None:
        alpha = N * b / (1 + (N * b))
        
    if len(indices)==0:
        print("Distance already calculated.")
        
    else: 
        
        worker_count = (mp.cpu_count()-1)
        
        manager = mp.Manager()
        queues = manager.Queue(2 * worker_count)
        
        partial_dist_workers_map_async = partial(dist_workers_map_async_partial, measure=meas, queue=queues)
        
        writer_procc = mp.Process(target=writer, args=(output_file_dir, queues, nof_operations), daemon=True)
        writer_procc.start()
        
        start = time.time()
        
        with mp.Pool(processes=mp.cpu_count()-1) as pool:

            pool.map_async(partial_dist_workers_map_async, iterator, chunksize=200)
                
            pool.close()
            pool.join()
        
        end = time.time()
        
        queues.put(None)
        writer_procc.join()

        print(f"Total duration is: {end-start} for {len(indices)} operations - map_async - partial pass")


def meas_dist_starmap_async(covs, mean=None, meas_name='TVD', alpha=None, b=1/100):
    
    N=len(covs) # maybe checking if covs is a numpy array of pickle file
    
    output_file_dir='/home/sezan/Documents/BayesCompare/parallel_tests_outs/parallel_tests_starmap_allayers.hdf5'    
    
    indices = check_saved_hdf(output_file_dir, N)
    
    nof_operations = len(indices)
    
    worker_count = (mp.cpu_count()-1)
    
    #cov_pairs = [(covs[i], covs[j]) for (i, j) in indices]
    
    meas = select_measure(meas_name)
    
    if alpha==None:
        alpha = N * b / (1 + (N * b))
        
    if len(indices)==0:
        print("Distance already calculated.")
        
    else: 
        
        manager = mp.Manager()
        queues = manager.Queue(2 * worker_count)
        
        writer_procc = mp.Process(target=writer, args=(output_file_dir, queues, nof_operations), daemon=True)
        writer_procc.start()
        
        task_list = zip(indices, repeat(covs), repeat(meas), repeat(queues))
        
        start = time.time()
        
        with mp.Pool(processes=mp.cpu_count()-1) as pool:

            pool.starmap_async(dist_workers_starmap_async, task_list)
                
            pool.close()
            pool.join()
        
        end = time.time()
        
        queues.put(None)
        writer_procc.join()

        print(f"Total duration is: {end-start} for {len(indices)} operations - starmap_async")

def non_parallel(covs):
    
    N = len(covs)   

    dist = np.zeros((len(covs), len(covs)))
    
    progress_bar = tqdm.tqdm(total=int((N*(N-1))/2))

    start = time.time()

    for i, ci in enumerate(covs):
        
        for j, cj in enumerate(covs):
            
            if j > i:
                
                dist[i, j] = wasserstein(ci, cj)
                
                dist[j, i] = dist[i, j]
                
                progress_bar.update(1)

    end = time.time()

    print(f"Total duration is: {end-start} for {int((N*(N-1))/2)} operations - normal")

if __name__ == "__main__":
    
    #input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_normalized.npy'
    input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_resnet50_densesampled_normalized.npy'
    #input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_all_resnets_all_layers_normalized.npy'
    
    covs = np.load(input_dir)
    
    ### Parallel approaches
    
    #meas_dist_map_async_global(covs, input_dir, meas_name='wasserstein')
    meas_dist_map_async_global_queue(covs, input_dir, meas_name='wasserstein')
    #meas_dist_map_async_partial(covs, input_dir, meas_name='wasserstein')
    #meas_dist_starmap_async(covs, input_dir, meas_name='wasserstein')
    
    ### Non-parallel approach
    
    #non_parallel(covs) 

# ----- Small Covs List - covs_1000.npy   -------

# Total duration is: 290.78909182548523 for 300 operations - map_async + global pass
# Total duration is: 296.02053475379944 for 300 operations - map_async + global queue pass
# Total duration is: 295.746386051178 for 300 operations - map_async - partial pass
# Total duration is: 307.9117214679718 for 300 operations - starmap_async

# Total duration is: 183.2870111465454 for 300 operations - non-parallel

# ----- Larger Covs List - densesampled   -------

# Estimated time by tqdm for map_async + global pass --- increments weirdly, not one by one 
# 1%|█▏                    | 231/37128 [03:30<1:55:01,  5.35it/s]

# Estimated time by tqdm for   --- **crashes after ~4mins**
# 0%|▊                     | 173/36910 [03:30<8:32:02,  1.20it/s

# Estimated time by tqdm for map_async + global queue pass --- 
# chunksize = 20
# 1%|█▎                    | 237/36935 [03:48<6:27:18,  1.58it/s]

# chunksize = 50
# 1%|█▏                    | 223/36674 [03:49<5:17:37,  1.91it/s]

# chunksize = 200
# 1%|█                     | 202/36417 [03:37<7:23:39,  1.36it/s]

# Estimated time by tqdm for map_async - partial pass --- **crashes after ~4mins**
# 0%|█                     | 175/37128 [03:33<10:01:17,  1.02it/s]

# chunksize = 20
# 1%|█▎                                   | 248/36899 [04:00<7:15:27,  1.40it/s]

# chunksize = 200 **crashed after ~9 mins**
# 1%|█▎                                   | 238/36538 [04:13<4:40:10,  2.16it/s]

# Estimated time by tqdm for map_async - starmap_async ---
# 1%|█▎                    | 206/37128 [03:34<15:36:41,  1.52s/it]
# Total duration is: 33459.614587306976 for 37128 operations - starmap_async -- 9:17:39

# Estimated time by tqdm for Normal computations
# 1%|█▊                    | 369/37128 [03:56<6:30:20,  1.57it/s]

# ----- Largest Covs List - alllayers   -------

# Estimated time by tqdm for map_async - starmap_async --- crashed

# Non-parallel
#   0%|▎                   | 431/386760 [04:25<69:46:20,  1.54it/s