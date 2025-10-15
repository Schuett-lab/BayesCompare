import tqdm
import numpy as np
import BayesCompare
import glob
import os

def save_checkpoint(dist, i, j):
    print("Saving checkpoint at i,j:", i, j)
    np.save("/home/sezan/Documents/BayesCompare/dist_checkpoints/checkpoint_dist_jsd_1000_resnet50_densesampled_"+str(i)+".npy", dist)
    

def load_checkpoint(checkpoint_dir):
    
    if os.path.exists(checkpoint_dir):
        
        checkpoint_path = checkpoint_dir + "checkpoint*.npy"
    
        matching_files = [f for f in glob.glob(checkpoint_path) if os.path.isfile(f)]
        
        if matching_files:
            max_i=0
            for filename in matching_files:
            
                saved_i = int(filename.split("_")[-1][:-4])

                if saved_i > max_i: 
                    max_i = saved_i
            
            latest_save_file = "/home/sezan/Documents/BayesCompare/dist_checkpoints/checkpoint_dist_jsd_1000_resnet50_densesampled_"+str(max_i)+".npy"
            
        else:
            dist = np.zeros((len(covs), len(covs)))
        
            return dist, 0
            
        
        print("Loading checkpoint from:", latest_save_file)
        
        dists = np.load(latest_save_file)
        
        return dists, int(max_i)
    
    else:
        dist = np.zeros((len(covs), len(covs)))
        
        return dist, 0
        
eye_w = 10/11

covs = np.load("/home/sezan/Documents/BayesCompare/covs_1000_all_resnets_densesampled.npy")

dist, i = load_checkpoint("/home/sezan/Documents/BayesCompare/dist_checkpoints/")

if i==0:
    for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
        for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
            if j > i:
                dist[i, j] = BayesCompare.jsd_normal_sig(ci, cj, 10000, eye_w=eye_w)
                dist[j, i] = dist[i, j]
                
        np.save("/home/sezan/Documents/BayesCompare/dist_checkpoints/checkpoint_dist_jsd_1000_resnet50_densesampled_"+str(i)+".npy", dist)
        
        
else:
    for i in tqdm.tqdm(range(i, len(covs)), initial=i, total=len(covs)):
        ci = covs[i]
        for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
            if j > i:
                dist[i, j] = BayesCompare.jsd_normal_sig(ci, cj, 10000, eye_w=eye_w)
                dist[j, i] = dist[i, j]
                
        np.save("/home/sezan/Documents/BayesCompare/dist_checkpoints/checkpoint_dist_jsd_1000_resnet50_densesampled_"+str(i)+".npy", dist)
     
np.save("/home/sezan/Documents/BayesCompare/dist_jsd_1000_resnet50_densesampled.npy", dist)