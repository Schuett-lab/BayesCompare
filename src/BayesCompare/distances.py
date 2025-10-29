import numpy as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import tqdm
import os
import glob
from joblib import Parallel, delayed, Memory

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
    

## Distance function caller

def measure_dist(covs, checkpoint_dir, mean=None, meas_name='TVD', alpha=None, b=1/100): # maybe set default alpha based on N (as in paper)
    
    # check if a single string or a list of strings is given in the meas_name
    if isinstance(meas_name, str):
        meas_name = [meas_name]
    
    N=len(covs)
    
    dist_dict = {}
    
    for name in meas_name:
        
        if alpha==None:
            alpha = N * b / (1 + (N * b))
            
        if name=='wasserstein':
            measure = wasserstein
            
        elif name=='hellinger':
            measure = hellinger
        
        elif name=='TVD':
            measure = tvd
        
        elif name=='JSD':
            measure = jsd
        
        elif name=='KL_div':
            measure = KL_div
            
        elif name=='bhattacharyya':
            measure = bhattacharyya
        
        dist, i = load_checkpoint(checkpoint_dir, name, N)
        
        if i==0:
        
            progress_bar = tqdm.tqdm(total=int((N*(N-1))/2))
            
            for i, ci in enumerate(covs):
                
                sig1 = trace_norm(ci, eye_w=alpha)
                
                for j, cj in enumerate(covs):
                    
                    if j > i:

                        sig2 = trace_norm(cj, eye_w=alpha)
                        
                        if measure == jsd or measure == tvd:
                            dist[i, j] = measure(sig1, sig2, N=10000) # not using mean, for a generalized code mean should be provided
                        else:
                            dist[i, j] = measure(sig1, sig2) # not using mean, for a generalized code mean should be provided
                            
                        dist[j, i] = dist[i, j]
                        
                        progress_bar.update(1)
                        
                save_checkpoint(dist, name, i, checkpoint_dir)
        
        else:
            
            already_calculated = 0
            
            for a in range(i):
                already_calculated += (N-1-a)
            
            progress_bar = tqdm.tqdm(initial=already_calculated, total=int((N*(N-1))/2))
            
            for i in range(i, N):
                
                ci = covs[i]
                
                sig1 = trace_norm(ci, eye_w=alpha)
                
                for j, cj in enumerate(covs):
                    
                    if j > i:

                        sig2 = trace_norm(cj, eye_w=alpha)
                        
                        if measure == jsd or measure == tvd:
                            dist[i, j] = measure(sig1, sig2, N=10000) # not using mean, for a generalized code mean should be provided
                        else:
                            dist[i, j] = measure(sig1, sig2) # not using mean, for a generalized code mean should be provided
                            
                        dist[j, i] = dist[i, j]
                        
                        progress_bar.update(1)
                        
                save_checkpoint(dist, name, i, checkpoint_dir)
        
        dist_dict[name] = dist
    
    # if a single measure is asked, return only a dist matrix
    if len(meas_name)==1:
        return dist_dict[meas_name[0]]
    
    # if multiple measures are asked, return a dict of dist matrices
    else:    
        return dist_dict            


def parallel_measure_dist(covs, checkpoint_dir, mean=None, meas_name='TVD', alpha=None, b=1/100, n_jobs=-1): # what should be n-jobs default?
    
    memory = Memory(location=checkpoint_dir, verbose=0)
    
    # check if a single string or a list of strings is given in the meas_name
    if isinstance(meas_name, str):
        meas_name = [meas_name]
    
    N=len(covs)
    
    dist_dict = {}
    
    norm_sigmas = Parallel(n_jobs=n_jobs, backend="loky")(delayed(trace_norm)(covs[i], alpha) for i in range(N))
    
    for name in meas_name:
        
        if alpha==None:
            alpha = N * b / (1 + (N * b))
            
        if name=='wasserstein':
            measure = wasserstein
            
        elif name=='hellinger':
            measure = hellinger
        
        elif name=='TVD':
            measure = tvd
        
        elif name=='JSD':
            measure = jsd
        
        elif name=='KL_div':
            measure = KL_div
            
        elif name=='bhattacharyya':
            measure = bhattacharyya
        
        @memory.cache
        def compute_pairwise_dist(i, j, meas):
            
            mi = norm_sigmas[i]
            mj = norm_sigmas[j]
            
            if measure in {jsd, tvd}:
                val = meas(mi, mj, N=5000)
            else:
                val = meas(mi, mj)
            return (i, j, val)
        
        upper_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        dist_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(delayed(compute_pairwise_dist)(i, j, measure) for (i, j) in upper_pairs)
        
        dist = np.zeros((N, N), dtype=float)
        
        for i, j, v in dist_list:
            dist[i, j] = v
            dist[j, i] = v
        
        dist_dict[name] = dist
    
    # if a single measure is asked, return only a dist matrix
    if len(meas_name)==1:
        return dist_dict[meas_name[0]]
    
    # if multiple measures are asked, return a dict of dist matrices
    else:    
        return dist_dict 


## Helper functions

def trace_norm(sigma, eye_w=0.001):
    
    if eye_w == 0:
        A = sigma
    
    else:
        A = ((1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)) + (eye_w * np.eye(sigma.shape[0]))
    
    return A

def save_checkpoint(dist, dist_name, i, directory):
    
    print("Saving checkpoint at i: ", str(i))
    print("Saving to the directory: " + directory + "checkpoint_dist_" + dist_name + "_" + str(i)+".npy")
    
    np.save(directory + "checkpoint_dist_" + dist_name + "_" + str(i)+".npy", dist)
    

def load_checkpoint(checkpoint_dir, dist_name, N):
    
    if os.path.exists(checkpoint_dir):
        
        checkpoint_path = checkpoint_dir + "checkpoint_dist_" + dist_name + "_*.npy"
    
        matching_files = [f for f in glob.glob(checkpoint_path) if os.path.isfile(f)]
        
        if matching_files:
            
            max_i=0
            
            for filename in matching_files:
            
                saved_i = int(filename.split("_")[-1][:-4])

                if saved_i > max_i: 
                    max_i = saved_i
            
            latest_save_file = checkpoint_dir + "checkpoint_dist_" + dist_name + "_" + str(max_i) + ".npy"
            
            print("Loading checkpoint from:", latest_save_file)
            
            dists = np.load(latest_save_file)
            
            return dists, int(max_i)+1
        
        # return dist=zeros and i=0 if there are no matching files    
        else:
            
            dist = np.zeros((N, N))
            
            return dist, 0
            
    # if a checkpoint path doesn't exist, create it and return dist=zeros and i=0
    else:
        
        os.mkdir(checkpoint_dir)
        
        dist = np.zeros((N, N))
        
        return dist, 0
