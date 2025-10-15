from BayesCompare import measure_dist
import numpy as np
import unittest
from itertools import pairwise

TOLERANCE = 1e-12

COV_CASES = {
    # 1. Well-conditioned (baseline SPD)
    "well": np.array([
        [2.0, 0.3, 0.2],
        [0.3, 1.5, 0.1],
        [0.2, 0.1, 1.0],
    ]),
 
    # 2. Ill-conditioned (tiny eigenvalue, nearly singular)
    "ill": np.array([
        [1.0, 0.99, 0.99],
        [0.99, 0.98, 0.97],
        [0.99, 0.97, 0.96],
    ]),
 
    # 3. Rank-deficient (singular PSD, rank 1)
    "rank_def": np.array([
        [1.0, 0.5, 0.5],
        [0.5, 0.25, 0.25],
        [0.5, 0.25, 0.25],
    ]),
 
    # 4a. Extreme scale: huge values
    "huge": np.array([
        [1e150, 0.0,    0.0],
        [0.0,    1e149, 0.0],
        [0.0,    0.0,   1e148],
    ]),
 
    # 4b. Extreme scale: tiny values
    "tiny": np.array([
        [1e-150, 0.0,      0.0],
        [0.0,    1e-151,   0.0],
        [0.0,    0.0,      1e-152],
    ]),
 
    # 5. Nearly PSD but not (tiny negative eigenvalue)
    "nearly_psd_not": np.array([
        [1.0,   0.99, 0.99],
        [0.99,  0.98, 0.97],
        [0.99,  0.97, 0.969],  # perturbed slightly downward
    ]),
 
    # 6. Non-symmetric (tests symmetrization safeguard)
    "non_sym": np.array([
        [1.0, 0.5, 0.0],
        [0.1, 1.0, 0.2],   # (0,1) ≠ (1,0)
        [0.0, 0.2, 1.0],
    ]),
 
    # 7. Mixed-scale block diagonal (4x4)
    "mixed_blocks": np.array([
        [1e8,   0.0,   0.0,    0.0],
        [0.0,   1e8,   0.0,    0.0],
        [0.0,   0.0,   1e-8,   0.0],
        [0.0,   0.0,   0.0,    1e-8],
    ]),
 
    # 8. High positive correlation (almost singular)
    "high_corr": np.array([
        [1.0, 0.9999, 0.9999],
        [0.9999, 1.0, 0.9999],
        [0.9999, 0.9999, 1.0],
    ]),
 
    # 9. High negative correlation (barely SPD)
    "neg_corr": np.array([
        [1.0, -0.999, -0.999],
        [-0.999, 1.0, -0.999],
        [-0.999, -0.999, 1.0],
    ]),
 
    # 10. Repeated eigenvalues (degenerate eigenspace)
    "repeated": np.array([
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0],
    ]),
}

class TestDistances(unittest.TestCase):
    
    def setUp(self):
        self.covs = COV_CASES
        self.mean1 = np.array([0.0, 0.0]) # zero means
        self.mean2 = np.array([1.0, 1.0]) # non-zero but equal means
        self.mean3 = np.array([1.2, 3.8]) # non-zero non-equal means
    
    def check_these(self, dist_matrix, cov_name1, cov_name2):
        msg = f"Distance between {cov_name1} and {cov_name2} failed checks"
        self.assertTrue(np.isfinite(dist_matrix).all(), msg=msg)
        self.assertAlmostEqual(dist_matrix[0, 1], dist_matrix[1, 0], msg=msg)
        #self.assertGreaterEqual(dist_matrix.all(), TOLERANCE, msg=msg)
        
    def test_wasserstein(self):
        
        for cov_name1, cov_name2 in pairwise(self.covs.keys()):
            dist_matrix = measure_dist([self.covs[cov_name1], self.covs[cov_name2]], meas_name='wasserstein')
            self.check_these(dist_matrix, cov_name1, cov_name2)
            
    def test_hellinger(self):
        
        for cov_name1, cov_name2 in pairwise(self.covs.keys()):
            dist_matrix = measure_dist([self.covs[cov_name1], self.covs[cov_name2]], meas_name='hellinger')
            self.check_these(dist_matrix, cov_name1, cov_name2)
    
    def test_matusita(self):
        
        for cov_name1, cov_name2 in pairwise(self.covs.keys()):
            dist_matrix = measure_dist([self.covs[cov_name1], self.covs[cov_name2]], meas_name='matusita')
            self.check_these(dist_matrix, cov_name1, cov_name2)
    
    def test_tvd(self):
        
        for cov_name1, cov_name2 in pairwise(self.covs.keys()):
            dist_matrix = measure_dist([self.covs[cov_name1], self.covs[cov_name2]], meas_name='TVD')
            self.check_these(dist_matrix, cov_name1, cov_name2)
    
    def test_jsd(self):
        
        for cov_name1, cov_name2 in pairwise(self.covs.keys()):
            dist_matrix = measure_dist([self.covs[cov_name1], self.covs[cov_name2]], meas_name='JSD')
            self.check_these(dist_matrix, cov_name1, cov_name2)
    
    def test_KL_div(self):
        
        for cov_name1, cov_name2 in pairwise(self.covs.keys()):
            dist_matrix = measure_dist([self.covs[cov_name1], self.covs[cov_name2]], meas_name='KL_div')
            self.check_these(dist_matrix, cov_name1, cov_name2)
        
    def test_bhattacharyya(self):
        
        for cov_name1, cov_name2 in pairwise(self.covs.keys()):
            dist_matrix = measure_dist([self.covs[cov_name1], self.covs[cov_name2]], meas_name='bhattacharyya')
            self.check_these(dist_matrix, cov_name1, cov_name2)

if __name__ == '__main__':
    unittest.main()