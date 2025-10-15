import numpy as np
 
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
 
if __name__ == "__main__":
    # Quick smoke test: print condition numbers
    for name, C in COV_CASES.items():
        try:
            cond = np.linalg.cond((C + C.T) * 0.5)
            print(f"{name:15s} cond ≈ {cond:.2e}")
        except np.linalg.LinAlgError:
            print(f"{name:15s} not invertible")