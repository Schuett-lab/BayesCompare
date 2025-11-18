import numpy as np
import h5py
import os
from joblib import Parallel, delayed
import multiprocessing as mp
from BayesCompare import distances as dist
import pickle
import tqdm


def check_saved_hdf(hdf_dir, N, covs_name, measure_name):

    hdf_filename = (
        os.path.join(hdf_dir, "") + "dist_" + covs_name + "_" + measure_name + ".hdf5"
    )

    if os.path.exists(hdf_filename):

        with h5py.File(hdf_filename, "r") as f:

            dist = f["dist"][...]

            tril_idx = np.tril_indices(dist.shape[0], k=-1)
            nan_mask = np.isnan(dist[tril_idx])
            indices = [
                [int(i), int(j)]
                for i, j in zip(tril_idx[0][nan_mask], tril_idx[1][nan_mask])
            ]

    else:

        with h5py.File(hdf_filename, "w") as f:

            init_mtx = np.empty((N, N)) * np.nan

            dist_dset = f.create_dataset("dist", shape=(N, N), data=init_mtx)
            indices = [(i, j) for j in range(N) for i in range(j + 1, N)]

            f.flush()

    return indices, hdf_filename


def check_normalization(covs, tolerance=1e-4):

    # randomly select one cov matrix from the list
    idx = np.random.randint(len(covs))

    trace_cov = covs[idx].trace()

    return abs(trace_cov - len(covs[idx])) < tolerance


def writer(file_dir, que, total_num_ops):

    progress_bar = tqdm.tqdm(total=int(total_num_ops))

    with h5py.File(file_dir, "r+") as f:

        res_dset = f["dist"]

        while 1:

            item = que.get()

            if item is None:
                break

            res_dset[item[0], item[1]] = item[2]
            res_dset[item[1], item[0]] = item[2]

            f.flush()

            progress_bar.update(1)


def load_covs(full_filename):

    _, ext = os.path.splitext(full_filename)

    pckl_exts = {".pkl", ".pickle"}
    numpy_exts = {".np", ".npy", ".npz"}

    if ext in pckl_exts:

        filename_ext = os.path.basename(full_filename)
        filename, ext = os.path.splitext(filename_ext)

        with open(full_filename, "rb") as f:
            covs_names = pickle.load(f)

        covs = []

        for cov_dict in covs_names:

            covs.append(list(cov_dict.values()))

        covs = np.stack(covs)
        covs = covs.reshape(covs.shape[0] * covs.shape[1], covs.shape[2], covs.shape[3])

    elif ext in numpy_exts:

        filename_ext = os.path.basename(full_filename)
        filename, ext = os.path.splitext(filename_ext)

        covs = np.load(full_filename)

    return covs, filename


def measure_dist_parallel(
    covs_dir,
    output_dir,
    mean=None,
    meas_name="TVD",
    num_workers=mp.cpu_count() - 1,
    alpha=None,
    b=1 / 100,
):

    # check if a single string or a list of strings is given in the meas_name
    if isinstance(meas_name, str):
        meas_name = [meas_name]

    covs, covs_filename = load_covs(covs_dir)

    assert check_normalization(
        covs
    ), "Invalid Operation: covariance matrices has to be trace normalized."

    N = len(covs)

    for name in meas_name:

        measure = dist.select_measure(name)

        indices, output_filename = check_saved_hdf(output_dir, N, covs_filename, name)

        if len(indices) == 0:
            print("Distance already calculated")

        else:

            with mp.Manager() as manager:

                output_queue = manager.Queue(2 * num_workers)

                writer_proc = mp.Process(
                    target=writer, args=(output_filename, output_queue, len(indices))
                )
                writer_proc.start()

                def pairwise_dist(i, j):

                    val = measure(covs[i], covs[j])
                    output_queue.put((i, j, val))

                Parallel(n_jobs=num_workers, backend="loky", verbose=0)(
                    delayed(pairwise_dist)(i, j) for i, j in indices
                )

                output_queue.put(None)
                writer_proc.join()
