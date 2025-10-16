import numpy as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import tqdm
import time


def sqrt_w_svd(A):
    
    lamb, u = np.linalg.eig(A) 
    sqroot = u @ np.eye(lamb.size)*np.sqrt(lamb) @ np.transpose(u)  
    
    return sqroot

def trace_norm(sigma, eye_w=0.001):
    
    if eye_w == 0:
        A = sigma
    
    else:
        A = ((1 - eye_w) * sigma * sigma.shape[0] / np.trace(sigma)) + (eye_w * np.eye(sigma.shape[0]))
    
    return A

'''
### Isolating Wasserstein and Testing it

# Wasserstein with Cholesky
def wasserstein1(sigma1, sigma2, mu1=None, mu2=None):
    
    # these conditions do not check for one mean is non zero and other is zero !!!
    if mu1 is not None and mu2 is not None:
        means_term = np.sum(np.square(mu1 - mu2))
    else:
        means_term = 0
    
    L1 = np.linalg.cholesky(sigma1)
    L121 = np.linalg.cholesky(L1 @ sigma2 @ L1.transpose())
    tr_term = sigma1 + sigma2 - 2*(L121) 
    d_sqd = means_term + np.trace(tr_term)
    
    return d_sqd**0.5

# Wasserstein with SVD
def wasserstein2(sigma1, sigma2, mu1=None, mu2=None):
    
    if mu1 is not None and mu2 is not None:
        means_term = np.linalg.norm(mu1 - mu2, 2)**2
    else:
        means_term=0
        
    sig1_sqrt = sqrt_w_svd(sigma1)
    sig1_sig2_sqrt = sqrt_w_svd(sig1_sqrt @ sigma2 @ sig1_sqrt)
    tr_term = sigma1 + sigma2 - 2*(sig1_sig2_sqrt) 
    d_sq = means_term + np.trace(tr_term)
    
    return d_sq**0.5

# Wasserstein with Sqrtm
def wasserstein3(sigma1, sigma2, mu1=None, mu2=None):
    
    if mu1 is not None and mu2 is not None:
        means_term = np.linalg.norm(mu1 - mu2, 2)**2
    else:
        means_term=0
        
    sig1_sqrt = scipy.linalg.sqrtm(sigma1)
    sig1_sig2_sqrt = scipy.linalg.sqrtm(sig1_sqrt @ sigma2 @ sig1_sqrt)
    tr_term = sigma1 + sigma2 - 2*(sig1_sig2_sqrt) 
    d_sq = means_term + np.trace(tr_term)
    
    return d_sq**0.5


covs = np.load("/home/sezan/Documents/BayesCompare/covs_1000.npy")

alpha = 10/11

dist1 = np.zeros((len(covs), len(covs)))
dist2 = np.zeros((len(covs), len(covs)))
dist3 = np.zeros((len(covs), len(covs)))

for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
    
    sig1 = trace_norm(ci, eye_w=alpha)
    
    for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
        
        if j > i:
            
            sig2 = trace_norm(cj, eye_w=alpha)
            
            #dist1[i, j] = wasserstein1(sig1, sig2)
            
            start1 = time.time()
            dist2[i, j] = wasserstein2(sig1, sig2)
            end1 = time.time()
            dist3[i, j] = wasserstein3(sig1, sig2)
            end2= time.time()
            
            print("SVD operation took: " + str(end1-start1))
            print("Sqrtm operation took: " + str(end2-end1))
            #dist1[j, i] = dist1[i, j]
            dist2[j, i] = dist2[i, j]
            dist3[j, i] = dist3[i, j]

## Results: 
### 1- Cholesky results are different from that of SVD and Sqrtm. SVD and Sqrtm are almost the same except for one entry in [23,24].
### 2- SVD takes about 2.5 seconds to compute while Sqrtm only takes about 0.5 seconds.
'''
