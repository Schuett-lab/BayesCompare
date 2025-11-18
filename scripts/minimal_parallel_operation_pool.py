import multiprocessing as mp
import h5py
import numpy  as np
import time

# version 1 - apply_async

'''def employ_workers(mtx_list, i, j, que):
    
    mi = mtx_list[i]
    mj = mtx_list[j]
    
    res = np.sum(mi)+np.sum(mj)
    
    que.put((i, j, res))
    

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
    
    output_file_dir = "/home/sezan/Documents/BayesCompare/parallel_ops.hdf5"
    
    m = 20 # number of matrices, one dim of output mtx
    n = 1000 # one dim of cov mtx
    
    max_dim = int((m*(m-1))/2)
    
    mtx_list = []
    
    for i in range(m):
        mtx_list.append(np.random.randn(n,n))
    
    manager = mp.Manager()
    queues = manager.Queue()
    
    writer_procc = mp.Process(target=writer, args=(output_file_dir, queues, max_dim), daemon=True)
    writer_procc.start()
    
    indices = [(i, j) for j in range(m) for i in range(j + 1, m)]
    
    start = time.time()
    
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        
        for (i, j) in indices:
            pool.apply_async(employ_workers, args=(mtx_list, i, j, queues))
            
        pool.close()
        pool.join()
        
    end = time.time()
    
    queues.put(None)
    writer_procc.join()
    
    print(f"Total duration is: {end-start} for {max_dim} operations - apply_async")'''

## Result: Total duration is: 47.65972399711609 for 190 operations - apply_async (m=20 & n=1000)

# version 2 - map async
'''
def init_vars(mtx_list, out_q):
    global G_Q, G_MTX
    G_Q = out_q
    G_MTX = mtx_list

def employ_workers(ind_tuple):
    
    i = ind_tuple[0]
    j = ind_tuple[1]
    mi = G_MTX[i]
    mj = G_MTX[j]
    
    res = np.sum(mi)+np.sum(mj)
    
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
    
    output_file_dir = "/home/sezan/Documents/BayesCompare/parallel_ops_v2.hdf5"
    
    mtx_list = []
    
    m = 300 # number of matrices, one dim of output mtx
    n = 1000 # one dim of cov mtx
    
    max_dim = int((m*(m-1))/2)
    
    for i in range(m):
        mtx_list.append(np.random.randn(n,n))
    
    manager = mp.Manager()
    queues = manager.Queue()
    
    writer_procc = mp.Process(target=writer, args=(output_file_dir, queues, max_dim), daemon=True)
    writer_procc.start()
    
    indices = [(i, j) for j in range(m) for i in range(j + 1, m)]
    
    start = time.time()
    
    with mp.Pool(processes=mp.cpu_count()-1, initializer=init_vars, initargs=(mtx_list, queues)) as pool:

        pool.map_async(employ_workers, indices)
            
        pool.close()
        pool.join()
    
    end = time.time()
      
    queues.put(None)
    writer_procc.join()
    
    print(f"Total duration is: {end-start} for {max_dim} operations - map_async")
'''   
## Result: Total duration is: 8.805874347686768 for 44850 operations - map_async (m=300 & n=1000)


# version 3- starmap_async
'''
def init_vars(mtx_list, out_q):
    global G_Q, G_MTX
    G_Q = out_q
    G_MTX = mtx_list

def employ_workers(i, j):
    
    mi = G_MTX[i]
    mj = G_MTX[j]
    
    res = np.sum(mi)+np.sum(mj)
    
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
    
    output_file_dir = "/home/sezan/Documents/BayesCompare/parallel_ops_v3.hdf5"
    
    mtx_list = []
    
    m = 300 # number of matrices, one dim of output mtx
    n = 1000 # one dim of cov mtx
    
    max_dim = int((m*(m-1))/2)
    
    for i in range(m):
        mtx_list.append(np.random.randn(n,n))
    
    manager = mp.Manager()
    queues = manager.Queue()
    
    writer_procc = mp.Process(target=writer, args=(output_file_dir, queues, max_dim), daemon=True)
    writer_procc.start()
    
    indices = [(i, j) for j in range(m) for i in range(j + 1, m)]
    
    start = time.time()
    
    with mp.Pool(processes=mp.cpu_count()-1, initializer=init_vars, initargs=(mtx_list, queues)) as pool:

        pool.starmap_async(employ_workers, indices)
            
        pool.close()
        pool.join()
    
    end = time.time()
      
    queues.put(None)
    writer_procc.join()
    
    print(f"Total duration is: {end-start} for {max_dim} operations - starmap_async")
'''
## Result: Total duration is: 8.858367204666138 for 44850 operations - starmap_async (m=300 & n=1000)



#filname = "/home/sezan/Documents/BayesCompare/parallel_tests_starmap_allayers.hdf5"
filname = "/home/sezan/Documents/BayesCompare/parallel_tests_outs/shared_memjoblib_small_1.hdf5"
import matplotlib.pyplot as plt

dist_mtx = np.zeros((25,25))

with h5py.File(filname, "r") as f:
    
    
    f.visititems(lambda name, obj: print(f"  {name}: {type(obj)}"))
    
    dset = f["dist"]
    data = dset[...]

    1+1
    # dset = f["results"]
    # idx_set = f["indices_todo"]
    
    # print("\nDataset info:")
    # print("  Shape:", dset.shape)
    # print("  Dtype:", dset.dtype)
    # print("  Attributes:", dict(dset.attrs))
    
    # data = dset[...]
    # idx = idx_set[...]
    
    # for dat in data:
    #     i = dat[0]
    #     j = dat[1]
    #     val = dat[2]
        
    #     dist_mtx[i, j] = val
    #     dist_mtx[j, i] = val
        
plt.figure()
plt.imshow(dist_mtx, "bone", vmin=0, vmax=np.max(dist_mtx))
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.show()