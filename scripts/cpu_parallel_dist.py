import multiprocessing as mp
import h5py
import numpy  as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import time
import os
import torch

output_file_dir='/home/sezan/Documents/BayesCompare/dists_parallel_wass.hdf5'

input_dir = '/home/sezan/Documents/BayesCompare/covs_1000_resnet50_densesampled.npy'

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


def init_vars(mtx_list, out_q, measure, alpha):
    global G_Q, G_MTX, G_MEAS, G_ALPHA
    G_Q = out_q
    G_MTX = mtx_list
    G_MEAS = measure
    G_ALPHA = alpha
    
def check_saved_hdf(hdf_dir, max_dim, N):
    
    if os.path.exists(hdf_dir):
        
        with h5py.File(hdf_dir, 'r') as f:
            
            indices_todo = f["indices_todo"][...]
    else:
        
        with h5py.File(hdf_dir, 'w') as f:
            
            indices_todo = np.array([(i, j) for j in range(N) for i in range(j + 1, N)])
        
            ind_dset = f.create_dataset("indices_todo", data=indices_todo)
            
            res_dset = f.create_dataset("results", shape=(max_dim,), dtype=np.dtype([('i', 'uint16'), ('j', 'uint16'), ('res', 'float32')]))
            
            f.flush()
         
    return indices_todo

def trace_norm(sigma, eye_w=0.001):
    
    if eye_w == 0:
        A = sigma
    
    else:
        A = ((1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)) + (eye_w * np.eye(sigma.shape[0]))
    
    return A

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
    

def dist_workers(ind_tuple):
    
    i = ind_tuple[0]
    j = ind_tuple[1]
    mi = G_MTX[i]
    mj = G_MTX[j]
    
    
    if G_MEAS in {jsd, tvd}:
        res = G_MEAS(trace_norm(mi, G_ALPHA), trace_norm(mj, G_ALPHA), N=5000)
    else:
        res = G_MEAS(trace_norm(mi, G_ALPHA), trace_norm(mj, G_ALPHA))
    
    G_Q.put((i, j, res))
    

def writer(file_dir, que, max_dim):
    
    with h5py.File(file_dir, 'r+') as f:
        
        res_dset = f["results"]
        ind_todo = f["indices_todo"]
        all_indices = np.array(ind_todo[...])
        
        last_len = max_dim - all_indices.shape[0]
        
        while 1:
            
            m = que.get()
            
            if m is None:
                break
            
            res_dset[last_len] = m
            done_idx = np.where((all_indices == np.array([m[0], m[1]])).all(axis=1))[0]
            all_indices = np.delete(all_indices, done_idx, axis=0)
            
            del f['indices_todo']
            f.create_dataset('indices_todo', data=all_indices)
            
            f.flush()
            last_len += 1


if __name__ == "__main__":
    
    meas_name = 'wasserstein'
    
    mtx_list = np.load(input_dir)
    
    N = len(mtx_list)
    
    manager = mp.Manager()
    queues = manager.Queue()
    
    max_dim = int((N*(N-1))/2)
    
    measure = select_measure(meas_name)
    
    alpha =10/11
    
    indices = check_saved_hdf(output_file_dir, max_dim, N)
    
    if len(indices[...])==0:
        print("Distance already calculated")
        
    else:    
        writer_procc = mp.Process(target=writer, args=(output_file_dir, queues, max_dim), daemon=True)
        writer_procc.start()
        
        start = time.time()
        
        with mp.Pool(processes=mp.cpu_count()-1, initializer=init_vars, initargs=(mtx_list, queues, measure, alpha)) as pool:

            pool.map_async(dist_workers, indices)
                
            pool.close()
            pool.join()
        
        end = time.time()
        
        queues.put(None)
        writer_procc.join()

        print(f"Total duration is: {end-start} for {max_dim} operations - map_async")


