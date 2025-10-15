import numpy as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import tqdm

## Metrics

def wasserstein(sigma1, sigma2, mu1=None, mu2=None):
    
    # these conditions do not check for one mean is non zero and other is zero !!!
    if mu1 is not None and mu2 is not None:
        means_term = np.sum(np.square(mu1 - mu2))
    else:
        means_term = 0
    
    L1 = np.linalg.cholesky(sigma1)
    L121 = np.linalg.cholesky(L1 @ sigma2 @ L1.transpose()) #not sure about L.T
    tr_term = sigma1 + sigma2 - 2*(L121) 
    d_sqd = means_term + np.trace(tr_term)
    
    return np.sqrt(d_sqd)

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

def measure_dist(covs, mean=None, meas_name='TVD', alpha=0.01): # maybe set default alpha based on N (as in paper)
    
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
    
    dist = np.zeros((len(covs), len(covs)))
    
    for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
        
        sig1 = trace_norm(ci, eye_w=alpha)
        
        for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
            
            if j > i:

                sig2 = trace_norm(cj, eye_w=alpha)
                
                if measure == jsd or measure == tvd:
                    dist[i, j] = measure(sig1, sig2, N=10000) # not using mean, for a generalized code mean should be provided
                else:
                    dist[i, j] = measure(sig1, sig2) # not using mean, for a generalized code mean should be provided
                    
                dist[j, i] = dist[i, j]
                
    return dist

## Helper functions

def trace_norm(sigma, eye_w=0.001):
    
    if eye_w == 0:
        A = sigma
    
    else:
        A = ((1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)) + (eye_w * np.eye(sigma.shape[0]))
    
    return A

def get_chols(sigma1, sigma2):
    
    A1 = np.linalg.cholesky(sigma1)
    A2 = np.linalg.cholesky(sigma2)
    A1_A2 = np.linalg.cholesky(sigma1 + sigma2)
    
    return 

'''
## Small cholesky linearity experiment:

A = np.array([[5, 3 ,2], [4, 6, 2], [1, 1, 8]])
La = np.linalg.cholesky(A)

B = np.array([[9, 3, 1], [3, 8, 5], [3, 3, 12]])
Lb = np.linalg.cholesky(B)

C = A + B
Lc = np.linalg.cholesky(C) # I suppose this may not even exist

Lc_hat = La + Lb

## Verdict: Lc_hat is not equal to Lc => Not linear!!
'''