import numpy as np
import scipy.linalg
from scipy.linalg.blas import dtrmm as mm
import tqdm
import time
import torch

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

def trace_norm_N(sigmas, eye_w=0.001):
    
    if eye_w == 0:
        A = sigmas
    
    else:
        A = ((1 - eye_w) * sigmas * sigmas.shape[-1] / sigmas.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)[:, None, None]) + (eye_w * torch.eye(sigmas.shape[-1])[None, ...])

    
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

## Comparing Wasserstein and Torch Compatible Wasserstein


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

def wasserstein_torch_comp(sigma1, sigma2, mu1=None, mu2=None):
    
    # these conditions do not check for one mean is non zero and other is zero !!!
    if mu1 is not None and mu2 is not None:
        means_term = np.linalg.norm(mu1 - mu2, 2)**2
    else:
        means_term=0
    
    if type(sigma1) != torch.Tensor:
        sigma1 = torch.tensor(sigma1)
    if type(sigma2) != torch.Tensor:
        sigma2 = torch.tensor(sigma2)
        
    E_sig1, V_sig1 = torch.linalg.eigh(sigma1)
    sig1_sqrt = (V_sig1 * torch.sqrt(E_sig1)) @ V_sig1.T
    
    sig12 = sig1_sqrt @ sigma2 @ sig1_sqrt
    E_sig12, V_sig12 = torch.linalg.eigh(sig12)
    sig1_sig2_sqrt = (V_sig12 * torch.sqrt(E_sig12)) @ V_sig12.T
    
    tr_term = sigma1 + sigma2 - 2*(sig1_sig2_sqrt) 
    d_sq = means_term + torch.trace(tr_term)
    
    if d_sq<0 and d_sq>-1e-7:
        d_sq = 0
    
    elif d_sq < -1e-7:
        raise ValueError(f"Wasserstein distance cannot be negative. Value is: {d_sq}")
    
    return d_sq**0.5

def wasserstein_torch(sigmas, mu1=None, mu2=None):
    
    # these conditions do not check for one mean is non zero and other is zero !!!
    if mu1 is not None and mu2 is not None:
        means_term = np.linalg.norm(mu1 - mu2, 2)**2
    else:
        means_term=0
    
    if type(sigmas) != torch.Tensor:
        sigmas = torch.tensor(sigmas) # N, B=2, D1, D2
       
    E_sig, V_sig = torch.linalg.eigh(sigmas[:, 0:1, :, :]) # E_sig = N, B, D1   V_sig = N, B, D1, D2
    sigs_sqrt = torch.einsum("NBDG, NBG, NBKG -> NBDK", V_sig, torch.sqrt(E_sig), V_sig)
    
    sig12 = torch.einsum("NBDK, NBKL, NBLM -> NBDM", sigs_sqrt, sigmas[:, 1:2, :, :], sigs_sqrt)
    E_sig12, V_sig12 = torch.linalg.eigh(sig12)
    sig12_sqrt = torch.einsum("NBDG, NBG, NBKG -> NBDK", V_sig12, torch.sqrt(E_sig12), V_sig12)
    
    tr_term = sigmas[:, 0:1, :, :] + sigmas[:, 1:2, :, :] - 2*sig12_sqrt
    d_sq = means_term + tr_term.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    d_sq = d_sq[:,0]
    # if d_sq<0 and d_sq>-1e-7:
    #     d_sq = 0
    
    # elif d_sq < -1e-7:
    #     raise ValueError(f"Wasserstein distance cannot be negative. Value is: {d_sq}")
    
    return d_sq**0.5

#covs = np.load("/home/sezan/Documents/BayesCompare/covs_1000_normalized.npy")

import pickle

with open('/home/sezan/Documents/BayesCompare/covs_1000_all_resnets_all_layers.pkl', "rb") as f:
    covs_names = pickle.load(f)

covs = []

for cov_dict in covs_names:
    
    covs.append(list(cov_dict.values()))
    layer_names = list(cov_dict.keys())

covs = np.stack(covs)
covs = covs.reshape(covs.shape[0] * covs.shape[1], covs.shape[2], covs.shape[3])

alpha = 10/11

dist1 = np.zeros((len(covs), len(covs)))
dist2 = np.zeros((len(covs), len(covs)))
dist3 = np.zeros((len(covs), len(covs)))
dist4 = np.zeros((len(covs), len(covs)))

N =len(covs)
N_total = int((N*(N-1))/2)

upper_pairs = [(i, j) for j in range(N) for i in range(j + 1, N)]

covs_subset = []

normed_covs = trace_norm_N(torch.Tensor(covs), eye_w=alpha)

# for i in range(25):
#     covs_subset.append([covs[upper_pairs[i]]])
    
# covs_subset = torch.Tensor(covs_subset)
# for i in range(N_total):
#     tens1 = torch.Tensor(normed_covs[upper_pairs[i][0]:upper_pairs[i][0]+1, :, :])
#     tens2 = torch.Tensor(normed_covs[upper_pairs[i][1]:upper_pairs[i][1]+1, :, :])
#     covs_subset.append(torch.concat((tens1, tens2), dim=0)[None, ...])
    
# covs_subset = torch.concat(covs_subset, dim=0)
    
# torch_out = wasserstein_torch(covs_subset)

# for i in range(N_total):
#     dist4[upper_pairs[i][0],upper_pairs[i][1]]=torch_out[i]
#     dist4[upper_pairs[i][1],upper_pairs[i][0]]=dist4[upper_pairs[i][0],upper_pairs[i][1]]

#for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
for i, ci in enumerate(covs):
      
    sig1 = trace_norm(ci, eye_w=alpha)
    
    #for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
    for j, cj in enumerate(covs):   
        if j > i:
            
            sig2 = trace_norm(cj, eye_w=alpha)
            
            dist1[i, j] = wasserstein(sig1, sig2)
            #dist2[i, j] = wasserstein_torch_comp(sig1, sig2)
            #dist3[i, j] = wasserstein_torch(np.concat((sig1[None, None, :, :], sig2[None, None, :, :]), axis=1))

            dist1[j, i] = dist1[i, j]
            #dist2[j, i] = dist2[i, j]
            #dist3[j, i] = dist3[i, j]

#diff_dist = dist1 - np.array(dist2)

#if abs(np.max(diff_dist)) < 1e-7:
#    print("Distances are the same!")

# Result: Indeed the two wasserstein computation return the same result.

# Comparing JSD and JSD torch compatable
'''
# GPU-capable generator (falls back to CPU if CUDA unavailable)
if torch.cuda.is_available():
    gen_torch = torch.Generator(device='cuda').manual_seed(42)
else:
    gen_torch = torch.Generator(device='cpu').manual_seed(42)

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

def jsd_torch_comp(sigma1, sigma2, mu1=None, mu2=None, N=10000, gen=gen_torch):
    
    if type(sigma1) != torch.Tensor:
        sigma1 = torch.tensor(sigma1)
    if type(sigma2) != torch.Tensor:
        sigma2 = torch.tensor(sigma2)
        
    if mu1 is None and mu2 is None:
                    
        k = sigma1.shape[0]
        A1 = torch.linalg.cholesky(sigma1)
        A2 = torch.linalg.cholesky(sigma2)
        logdet1 = torch.sum(torch.log(torch.diag(A1)))
        logdet2 = torch.sum(torch.log(torch.diag(A2)))
        # generate random samples from each distribution
        x10 = torch.randn((k, N), generator=gen_torch)
        x1 = torch.Tensor(mm(1, A1, x10, lower=1))
        x20 = torch.randn((k, N), generator=gen_torch)
        x2 = torch.Tensor(mm(1, A2, x20, lower=1))
        # compute densities for each
        p1 = - torch.sum(x10 ** 2, 0) / 2 - logdet1
        delta21 = torch.linalg.solve_triangular(A1, x2, upper=False)
        p2 = - torch.sum(delta21 ** 2, 0) / 2 - logdet1
        delta12 = torch.linalg.solve_triangular(A2, x1, upper=False)
        q1 = - torch.sum(delta12 ** 2, 0) / 2 - logdet2
        q2 = - torch.sum(x20 ** 2, 0) / 2 - logdet2

        # log (P) - log (P + Q)
        term1 = p1 - torch.logaddexp(p1, q1)
        term2 = q2 - torch.logaddexp(p2, q2)

        jsd = 1 + (torch.mean(term1) + torch.mean(term2)) / 2 / np.log(2)
        
        
    else:
        
        k = len(mu1)
        A1 = torch.linalg.cholesky(sigma1)
        A2 = torch.linalg.cholesky(sigma2)

        Ainv1 = torch.linalg.solve_triangular(A1, torch.eye(k), upper=False)
        Ainv2 = torch.linalg.solve_triangular(A2, torch.eye(k), upper=False)
        # generate random samples from each distribution
        x1 = torch.Tensor.expand(mu1, 1) + A1 @ gen.standard_normal(size=(k, N))
        x2 = torch.Tensor.expand(mu2, 1) + A2 @ gen.standard_normal(size=(k, N))
        # compute densities for each
        # removed factor 2 from these as it cancels
        logdet1 = torch.sum(torch.log(torch.diag(A1)))
        logdet2 = torch.sum(torch.log(torch.diag(A2)))
        delta11 = Ainv1 @ (x1 - torch.Tensor.expand(mu1, 1))
        p1 = - torch.sum(delta11 ** 2, 0) / 2 - logdet1
        delta21 = Ainv1 @ (x2 - torch.Tensor.expand(mu1, 1))
        p2 = - torch.sum(delta21 ** 2, 0) / 2 - logdet1
        delta12 = Ainv2 @ (x1 - torch.Tensor.expand(mu2, 1))
        q1 = - torch.sum(delta12 ** 2, 0) / 2 - logdet2
        delta22 = Ainv2 @ (x2 - torch.Tensor.expand(mu2, 1))
        q2 = - torch.sum(delta22 ** 2, 0) / 2 - logdet2

        # log (P) - log (P + Q)
        term1 = p1 - torch.logaddexp(p1, q1)
        term2 = q2 - torch.logaddexp(p2, q2)

        jsd = 1 + (torch.mean(term1) + torch.mean(term2)) / 2 / torch.log(2)
        
    return max(0, jsd) 

covs = np.load("/home/sezan/Documents/BayesCompare/covs_1000.npy")

alpha = 10/11

dist1 = np.zeros((len(covs), len(covs)))
dist2 = np.zeros((len(covs), len(covs)))

for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
    
    sig1 = trace_norm(ci, eye_w=alpha)
    
    for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
        
        if j > i:
            
            sig2 = trace_norm(cj, eye_w=alpha)
            
            dist1[i, j] = jsd(sig1, sig2)
            dist2[i, j] = jsd_torch_comp(sig1, sig2)

            dist1[j, i] = dist1[i, j]
            dist2[j, i] = dist2[i, j]

diff_dist = dist1 - np.array(dist2)

if abs(np.max(diff_dist)) < 1e-7:
    print("Distances are the same!")
'''    
# Results: The maximum difference between JSD and torch compatible JSD is actually 0.0198 which is not so much negligible.


# Comparing TVD and TVD torch compatable
'''
if torch.cuda.is_available():
    gen_torch = torch.Generator(device='cuda').manual_seed(42)
else:
    gen_torch = torch.Generator(device='cpu').manual_seed(42)

gen = np.random.Generator(np.random.SFC64(42))

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

def tvd_torch_comp(sigma1, sigma2, mu1=None, mu2=None, N=10000, gen=gen_torch):
    
    if type(sigma1) != torch.Tensor:
        sigma1 = torch.tensor(sigma1)
    if type(sigma2) != torch.Tensor:
        sigma2 = torch.tensor(sigma2)

    if mu1 is not None and mu2 is not None:
        k = len(mu1)
        A1 = torch.linalg.cholesky(sigma1)
        A2 = torch.linalg.cholesky(sigma2)
        Ainv1 = torch.linalg.solve_triangular(A1, torch.eye(k), upper=False)
        Ainv2 = torch.linalg.solve_triangular(A2, torch.eye(k), upper=False)
        # generate random samples from each distribution
        x1 = torch.Tensor.expand(mu1, 1) + A1 @ torch.randn((k, N), generator=gen)
        x2 = torch.Tensor.expand(mu2, 1) + A2 @ torch.randn((k, N), generator=gen)
        # compute densities for each
        # removed factor 2 from these as it cancels
        logdet1 = torch.sum(torch.log(torch.diag(A1)))
        logdet2 = torch.sum(torch.log(torch.diag(A2)))
        delta11 = Ainv1 @ (x1 - torch.Tensor.expand(mu1, 1))
        p1 = - torch.sum(delta11 ** 2, 0) / 2 - logdet1
        delta21 = Ainv1 @ (x2 - torch.Tensor.expand(mu1, 1))
        p2 = - torch.sum(delta21 ** 2, 0) / 2 - logdet1
        delta12 = Ainv2 @ (x1 - torch.Tensor.expand(mu2, 1))
        q1 = - torch.sum(delta12 ** 2, 0) / 2 - logdet2
        delta22 = Ainv2 @ (x2 - torch.Tensor.expand(mu2, 1))
        q2 = - torch.sum(delta22 ** 2, 0) / 2 - logdet2
        f1 = max(1 - torch.exp(q1-p1), 0)
        f2 = max(1 - torch.exp(p2-q2), 0)
        tvd = (torch.mean(f1) + torch.mean(f2)) / 2
        
    else:
        k = sigma1.shape[0]
        A1 = torch.linalg.cholesky(sigma1)
        A2 = torch.linalg.cholesky(sigma2)
        logdet1 = torch.sum(torch.log(torch.diag(A1)))
        logdet2 = torch.sum(torch.log(torch.diag(A2)))
        # generate random samples from each distribution
        x10 = torch.randn((k, N), generator=gen)
        x1 = torch.Tensor(mm(1, A1, x10, lower=1))
        x20 = torch.randn((k, N), generator=gen)
        x2 = torch.Tensor(mm(1, A2, x20, lower=1))
        # compute densities for each
        p1 = - torch.sum(x10 ** 2, 0) / 2 - logdet1
        delta21 = torch.linalg.solve_triangular(A1, x2, upper=False)
        p2 = - torch.sum(delta21 ** 2, 0) / 2 - logdet1
        delta12 = torch.linalg.solve_triangular(A2, x1, upper=False)
        q1 = - torch.sum(delta12 ** 2, 0) / 2 - logdet2
        q2 = - torch.sum(x20 ** 2, 0) / 2 - logdet2
        f1 = torch.max(1 - torch.exp(q1-p1), torch.zeros_like(q1))
        f2 = torch.max(1 - torch.exp(p2-q2), torch.zeros_like(p2))
        tvd = (torch.mean(f1) + torch.mean(f2)) / 2
    
    return max(0, tvd)

covs = np.load("/home/sezan/Documents/BayesCompare/covs_1000.npy")

alpha = 10/11

dist1 = np.zeros((len(covs), len(covs)))
dist2 = np.zeros((len(covs), len(covs)))

for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
    
    sig1 = trace_norm(ci, eye_w=alpha)
    
    for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
        
        if j > i:
            
            sig2 = trace_norm(cj, eye_w=alpha)
            
            dist1[i, j] = tvd(sig1, sig2)
            dist2[i, j] = tvd_torch_comp(sig1, sig2)

            dist1[j, i] = dist1[i, j]
            dist2[j, i] = dist2[i, j]

diff_dist = dist1 - np.array(dist2)

if abs(np.max(diff_dist)) < 1e-7:
    print("Distances are the same!")
    '''
# Results: The maximum difference between TVD and torch compatible TVD is actually 0.012022903240433647 which is not so much negligible.