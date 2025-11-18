import numpy as np
import BayesCompare
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import PIL
import BayesCompare
from torchvision import transforms as tvt
import torch
import torchvision
import tqdm
from sklearn.manifold import MDS
import scipy
from torchvision.models.feature_extraction import get_graph_node_names
import pickle

###------------------------------- For only Wasserstein and a subset of layers of 5 ResNets.
'''
## Compute the covariances from trained models:

# Load a checkpoint model
snapshot1 = torch.load("/home/sezan/Documents/BayesCompare/checkpoints/snapshot_ep99_seed128_iter7.pth")
snapshot2 = torch.load("/home/sezan/Documents/BayesCompare/checkpoints/snapshot_ep99_seed333_iter8.pth")
snapshot3 = torch.load("/home/sezan/Documents/BayesCompare/checkpoints/snapshot_ep200_seed122_iter6.pth")
snapshot4 = torch.load("/home/sezan/Documents/BayesCompare/checkpoints/snapshot_ep200_seed400_iter4.pth")

model1 = torchvision.models.get_model("resnet50", weights=None)
model2 = torchvision.models.get_model("resnet50", weights=None)
model3 = torchvision.models.get_model("resnet50", weights=None)
model4 = torchvision.models.get_model("resnet50", weights=None)
model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

model1.load_state_dict(snapshot1["model_state"])
model2.load_state_dict(snapshot2["model_state"])
model3.load_state_dict(snapshot3["model"])
model4.load_state_dict(snapshot4["model"])

models = [model1, model2, model3, model4, model5]

for model in models:
    model.eval()

# Load the MS COCO dataset
im_folder = '/home/sezan/Documents/BayesCompare/images/unlabeled2017'
file_names = os.listdir(im_folder)

N = 1000
ims = [PIL.Image.open(
    os.path.join(im_folder, f_name))
    for f_name in file_names[:N]
]

interpolation=tvt.functional.InterpolationMode.BILINEAR
transform = tvt.Compose([tvt.Resize(232, interpolation=interpolation, antialias=True),
                         tvt.CenterCrop(224),
                         tvt.ToTensor(),
                         tvt.Normalize(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                         )
                        ])

transformed_ims = [transform(im.convert('RGB')) for im in ims]
x_input = torch.stack(transformed_ims)

# check the layer names with train_nodes before getting the features
#train_nodes1, _ = get_graph_node_names(model1)
                
set_1 = ["conv1", "bn1", "relu", "maxpool",
    "layer1.0.conv1", "layer1.0.bn1", "layer1.0.relu"]

set_2 =["layer1.0.conv3", "layer1.0.bn3", "layer1.0.downsample.0", "layer1.0.downsample.1", "layer1.0.add", "layer1.0.relu_2",
    "layer1.1.conv3", "layer1.1.bn3", "layer1.1.add", "layer1.1.relu_2"]

set_3 = ["layer1.2.conv1", "layer1.2.bn1", "layer1.2.relu",
    "layer2.0.conv1", "layer2.0.bn1", "layer2.0.relu",
    "layer2.0.conv3", "layer2.0.bn3", "layer2.0.downsample.0", "layer2.0.downsample.1", "layer2.0.add", "layer2.0.relu_2"]

set_4 = ["layer2.2.conv1", "layer2.2.bn1", "layer2.2.relu",
    "layer2.2.conv3", "layer2.2.bn3", "layer2.2.add", "layer2.2.relu_2",
    "layer2.3.conv1", "layer2.3.bn1", "layer2.3.relu"]

set_5 = ["layer2.3.conv3", "layer2.3.bn3", "layer2.3.add", "layer2.3.relu_2",
    "layer3.0.conv3", "layer3.0.bn3", "layer3.0.downsample.0", "layer3.0.downsample.1", "layer3.0.add", "layer3.0.relu_2"]

set_6 = ["layer3.1.conv2", "layer3.1.bn2", "layer3.1.relu_1", "layer3.1.conv3", "layer3.1.bn3", "layer3.1.add", "layer3.1.relu_2",
    "layer3.2.conv3", "layer3.2.bn3", "layer3.2.add", "layer3.2.relu_2"]

set_7 = ["layer3.3.conv1", "layer3.3.bn1", "layer3.3.relu",
    "layer3.4.conv1", "layer3.4.bn1", "layer3.4.relu",
    "layer3.4.conv2", "layer3.4.bn2", "layer3.4.relu_1"]

set_8 = ["layer3.4.conv3", "layer3.4.bn3", "layer3.4.add", "layer3.4.relu_2",
    "layer3.5.conv3", "layer3.5.bn3", "layer3.5.add", "layer3.5.relu_2",
    "layer4.0.conv1", "layer4.0.bn1", "layer4.0.relu"]

set_9 = ["layer4.0.conv3", "layer4.0.bn3", "layer4.0.downsample.0", "layer4.0.downsample.1", "layer4.0.add",
    "layer4.2.conv3", "layer4.2.bn3", "layer4.2.add", "layer4.2.relu_2",
    "avgpool", "fc"]

sets = [set_1, set_2, set_3, set_4, set_5, set_6, set_7, set_8, set_9]

model_num = 1

set_num = 1

covs = []

with torch.inference_mode():
    
    for model in tqdm.tqdm(models, position=0, desc="Models"):
        
        for set in tqdm.tqdm(sets, position=1, desc="Layer Sets"):
            
            covs_extractor = BayesCompare.cov_extractor(model, return_nodes=set)
            
            covs = covs_extractor(x_input)

np.save("covs_1000_all_resnets_densesampled.npy", np.stack(covs))
'''

## Directly from saved covariance matrix

'''
covs = np.load("/home/sezan/Documents/BayesCompare/covs_1000_all_resnets_densesampled.npy")

dist = BayesCompare.measure_dist(covs, checkpoint_dir="/home/sezan/Documents/BayesCompare/my_dist_checkpoints/", meas_name=['wasserstein'], alpha=10/11)

dist = dist['wasserstein']

np.save("dist_wasserstein_corrected_1000_all_resnets_densesampled.npy", dist)

'''
## Directly from saved distance matrix

'''
dist = np.load("nonan_symm_dist_wasserstein_corrected_1000_all_resnets_densesampled.npy")

plt.figure()
plt.imshow(dist, "bone", vmin=0, vmax=np.max(dist))
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.savefig("figures/dist_wasserstein_corrected_allresnets.svg", dpi=600)

normalized_dists = dist/(np.max(dist)) # since minimum is 0

plt.figure()
plt.imshow(normalized_dists, "bone", vmin=0, vmax=1)
plt.colorbar()
ax = plt.gca()
ax.set_axis_off()
plt.savefig("figures/normalized_dist_wasserstein_corrected_allresnets.svg", dpi=600)


mds = MDS(dissimilarity="precomputed")
mds.fit(dist)

n = 91 # number of layers per model
m = 5 # number of models

x = mds.embedding_
x0 = x[1:n+1]
x1 = x[n+2:2*n+2]
x2 = x[2*n+3:3*n+3]
x3 = x[3*n+4:4*n+4]
x4 = x[4*n+5:5*n+5]

plt.figure()
plt.plot(x0[:, 0], x0[:, 1], '.-', linewidth=2, markersize=10, color="#fa5750")  
plt.plot(x1[:, 0], x1[:, 1], '.-', linewidth=2, markersize=10, color="#dbb32d")  
plt.plot(x2[:, 0], x2[:, 1], '.-', linewidth=2, markersize=10, color="#4695f7")  
plt.plot(x3[:, 0], x3[:, 1], '.-', linewidth=2, markersize=10, color="#33db2d") 
plt.plot(x4[:, 0], x4[:, 1], '.-', linewidth=2, markersize=10, color="#d446f7") 
plt.arrow(
    x[0, 0], x[0, 1], x0[0, 0] - x[0, 0], x0[0, 1] - x[0, 1],
    length_includes_head=True, width=0.01, fc='black', ec=None)
plt.arrow(
    x[0, 0], x[0, 1], x1[0, 0] - x[0, 0], x1[0, 1] - x[0, 1],
    length_includes_head=True, width=0.01, fc='black', ec=None)
plt.arrow(
    x[0, 0], x[0, 1], x2[0, 0] - x[0, 0], x2[0, 1] - x[0, 1],
    length_includes_head=True, width=0.01, fc='black', ec=None)
plt.arrow(
    x[0, 0], x[0, 1], x3[0, 0] - x[0, 0], x3[0, 1] - x[0, 1],
    length_includes_head=True, width=0.01, fc='black', ec=None)
plt.arrow(
    x[0, 0], x[0, 1], x4[0, 0] - x[0, 0], x4[0, 1] - x[0, 1],
    length_includes_head=True, width=0.01, fc='black', ec=None)
plt.axis("equal")
ax = plt.gca()
ax.set_axis_off()
plt.savefig("figures/resnet50_results/normalized_dist_wasserstein_corrected_allresnets_densesampled_mds.svg")
plt.savefig("figures/resnet50_results/normalized_dist_wasserstein_corrected_allresnets_densesampled_mds.pdf")

plt.figure(figsize=(14,6), dpi=400)

blocks = dist.reshape((m, n, m, n)).transpose(0,2,1,3).reshape(m*m,n,n)
all_idx = np.triu(np.arange(m*m).reshape(m,m),k=1)
idx = all_idx[np.where(all_idx!=0)]

model_names = ["1 2", "1 3", "1 4", "1 5", "2 3", "2 4", "2 5", "3 4", "3 5", "4 5"]

model_idx = 0

cmap = plt.cm.get_cmap('viridis', 10)

model_layers = [
    "conv1", "bn1", "relu", "maxpool",
    "layer1.0.conv1", "layer1.0.bn1", "layer1.0.relu",
    "layer1.0.conv3", "layer1.0.bn3", "layer1.0.downsample.0", "layer1.0.downsample.1", "layer1.0.add", "layer1.0.relu_2",
    "layer1.1.conv3", "layer1.1.bn3", "layer1.1.add", "layer1.1.relu_2",
    "layer1.2.conv1", "layer1.2.bn1", "layer1.2.relu",
    "layer2.0.conv1", "layer2.0.bn1", "layer2.0.relu",
    "layer2.0.conv3", "layer2.0.bn3", "layer2.0.downsample.0", "layer2.0.downsample.1", "layer2.0.add", "layer2.0.relu_2",
    "layer2.2.conv1", "layer2.2.bn1", "layer2.2.relu",
    "layer2.2.conv3", "layer2.2.bn3", "layer2.2.add", "layer2.2.relu_2",
    "layer2.3.conv1", "layer2.3.bn1", "layer2.3.relu",
    "layer2.3.conv3", "layer2.3.bn3", "layer2.3.add", "layer2.3.relu_2",
    "layer3.0.conv3", "layer3.0.bn3", "layer3.0.downsample.0", "layer3.0.downsample.1", "layer3.0.add", "layer3.0.relu_2",
    "layer3.1.conv2", "layer3.1.bn2", "layer3.1.relu_1", "layer3.1.conv3", "layer3.1.bn3", "layer3.1.add", "layer3.1.relu_2",
    "layer3.2.conv3", "layer3.2.bn3", "layer3.2.add", "layer3.2.relu_2",
    "layer3.3.conv1", "layer3.3.bn1", "layer3.3.relu",
    "layer3.4.conv1", "layer3.4.bn1", "layer3.4.relu",
    "layer3.4.conv2", "layer3.4.bn2", "layer3.4.relu_1",
    "layer3.4.conv3", "layer3.4.bn3", "layer3.4.add", "layer3.4.relu_2",
    "layer3.5.conv3", "layer3.5.bn3", "layer3.5.add", "layer3.5.relu_2",
    "layer4.0.conv1", "layer4.0.bn1", "layer4.0.relu",
    "layer4.0.conv3", "layer4.0.bn3", "layer4.0.downsample.0", "layer4.0.downsample.1", "layer4.0.add",
    "layer4.2.conv3", "layer4.2.bn3", "layer4.2.add", "layer4.2.relu_2",
    "avgpool", "fc"
]

stack_diags = np.zeros((int(m*(m-1)/2), n))

for i in idx:
    diags = np.diag(blocks[i])
    stack_diags[model_idx, :] = diags
    plt.plot(range(n), diags, '.-', linewidth=2, markersize=5, label="Model "+model_names[model_idx][0]+" vs Model "+model_names[model_idx][2], color=cmap(model_idx))
    model_idx += 1
    
plt.xlabel("Layer")
plt.xticks(ticks=range(n), labels=model_layers, rotation=90, fontsize=6)
plt.grid(axis='x', color='gray', alpha=0.3, linewidth=0.5)
plt.ylabel("Wasserstein Distance")
plt.legend()
plt.tight_layout()
plt.savefig("figures/resnet50_results/dist_wasserstein_corrected_allresnets_densesampled_retrieval.svg", dpi=400)


mean = stack_diags.mean(axis=0)
var = stack_diags.var(axis=0, ddof=1)
std = np.sqrt(var)

plt.figure(figsize=(14, 6), dpi=400)

plt.plot(range(n), mean, '-o', markersize=2, linewidth=1, color='black')
plt.fill_between(range(n), mean-std, mean+std, alpha=0.15, color='gray')
plt.xticks(ticks=range(n), labels=model_layers, rotation=90, fontsize=6)
plt.grid(axis='x', color='gray', alpha=0.1, linewidth=0.5)
plt.xlabel("Layer")
plt.ylabel("Wasserstein Distance")
plt.tight_layout()
plt.savefig("figures/resnet50_results/avg_dist_wasserstein_corrected_corrected_allresnets_densesampled_retrieval.svg", dpi=400)
'''

###------------------------------- For Wasserstein and JSD and for all layers of all ResNets

# Load a checkpoint model
snapshot1 = torch.load("/home/sezan/Documents/BayesCompare/checkpoints/snapshot_ep99_seed128_iter7.pth")
snapshot2 = torch.load("/home/sezan/Documents/BayesCompare/checkpoints/snapshot_ep99_seed333_iter8.pth")
snapshot3 = torch.load("/home/sezan/Documents/BayesCompare/checkpoints/snapshot_ep200_seed122_iter6.pth")
snapshot4 = torch.load("/home/sezan/Documents/BayesCompare/checkpoints/snapshot_ep200_seed400_iter4.pth")

model1 = torchvision.models.get_model("resnet50", weights=None)
model2 = torchvision.models.get_model("resnet50", weights=None)
model3 = torchvision.models.get_model("resnet50", weights=None)
model4 = torchvision.models.get_model("resnet50", weights=None)
model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

model1.load_state_dict(snapshot1["model_state"])
model2.load_state_dict(snapshot2["model_state"])
model3.load_state_dict(snapshot3["model"])
model4.load_state_dict(snapshot4["model"])

models = [model1, model2, model3, model4, model5]

for model in models:
    model.eval()

# Load the MS COCO dataset
im_folder = '/home/sezan/Documents/BayesCompare/images/unlabeled2017'
file_names = os.listdir(im_folder)

N = 1000
ims = [PIL.Image.open(
    os.path.join(im_folder, f_name))
    for f_name in file_names[:N]
]

interpolation=tvt.functional.InterpolationMode.BILINEAR
transform = tvt.Compose([tvt.Resize(232, interpolation=interpolation, antialias=True),
                         tvt.CenterCrop(224),
                         tvt.ToTensor(),
                         tvt.Normalize(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                         )
                        ])

transformed_ims = [transform(im.convert('RGB')) for im in ims]
x_input = torch.stack(transformed_ims)

# check the layer names with train_nodes before getting the features
node_names, _ = get_graph_node_names(model1)

covs = []

with torch.inference_mode():
    
    for model in tqdm.tqdm(models):
        
        covs_extractor = BayesCompare.cov_extractor(model, return_nodes=node_names)        
        covs.append(covs_extractor(x_input))

with open('covs_1000_all_resnets_all_layers.pkl', "wb") as f:
    pickle.dump(covs, f)


## Directly from saved covariance matrix
'''
with open('covs_1000_all_resnets_all_layers.pkl', "rb") as f:
    covs_names = pickle.load(f)

covs = []

for cov_dict in covs_names:
    
    covs.append(list(cov_dict.values()))
    layer_names = list(cov_dict.keys())

covs = np.stack(covs)
covs = covs.reshape(covs.shape[0] * covs.shape[1], covs.shape[2], covs.shape[3])

dist = BayesCompare.measure_dist(covs, checkpoint_dir="/home/sezan/Documents/BayesCompare/dist_checkpoints/", meas_name=['wasserstein', 'JSD'], alpha=10/11)

np.save("dist_wasserstein_1000_all_resnets_all_layers.npy", dist['wasserstein'])

np.save("dist_JSD_1000_all_resnets_all_layers.npy", dist['JSD'])
'''
## Directly from saved distances
'''
means = []
stds = []

main_colors = ['steelblue', 'indigo']
error_colors = ['powderblue', 'thistle']

dist_names = ['wasserstein', 'JSD']

for dist_name in dist_names:
    
    dist = np.load("dist_"+dist_name+"_1000_all_resnets_all_layers.npy")

    plt.figure()
    plt.imshow(dist, "bone", vmin=0, vmax=np.max(dist))
    plt.colorbar()
    plt.title(dist_name.capitalize()+" Distance")
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig("figures/resnet50_results/all_layers/dist_"+dist_name+"_allresnets_alllayers.svg", dpi=800)

    n = int(len(dist)/5) # number of layers per model
    m = 5 # number of models
    

    mds = MDS(dissimilarity="precomputed")
    mds.fit(dist)
    
    
    x = mds.embedding_
    x0 = x[1:n+1, :]
    x1 = x[n+2:2*n+2, :]
    x2 = x[2*n+3:3*n+3, :]
    x3 = x[3*n+4:4*n+4, :]
    x4 = x[4*n+5:5*n+5, :]

    plt.figure()
    plt.plot(x0[:, 0], x0[:, 1], '.-', linewidth=2, markersize=10, color="#fa5750")  
    plt.plot(x1[:, 0], x1[:, 1], '.-', linewidth=2, markersize=10, color="#dbb32d")  
    plt.plot(x2[:, 0], x2[:, 1], '.-', linewidth=2, markersize=10, color="#4695f7")  
    plt.plot(x3[:, 0], x3[:, 1], '.-', linewidth=2, markersize=10, color="#33db2d") 
    plt.plot(x4[:, 0], x4[:, 1], '.-', linewidth=2, markersize=10, color="#d446f7") 
    plt.arrow(
        x[0, 0], x[0, 1], x0[0, 0] - x[0, 0], x0[0, 1] - x[0, 1],
        length_includes_head=True, width=0.01, fc='black', ec=None)
    plt.arrow(
        x[0, 0], x[0, 1], x1[0, 0] - x[0, 0], x1[0, 1] - x[0, 1],
        length_includes_head=True, width=0.01, fc='black', ec=None)
    plt.arrow(
        x[0, 0], x[0, 1], x2[0, 0] - x[0, 0], x2[0, 1] - x[0, 1],
        length_includes_head=True, width=0.01, fc='black', ec=None)
    plt.arrow(
        x[0, 0], x[0, 1], x3[0, 0] - x[0, 0], x3[0, 1] - x[0, 1],
        length_includes_head=True, width=0.01, fc='black', ec=None)
    plt.arrow(
        x[0, 0], x[0, 1], x4[0, 0] - x[0, 0], x4[0, 1] - x[0, 1],
        length_includes_head=True, width=0.01, fc='black', ec=None)
    plt.axis("equal")
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig("figures/resnet50_results/all_layers/dist_"+dist_name+"_allresnets_alllayers_mds.svg")
    plt.savefig("figures/resnet50_results/all_layers/dist_"+dist_name+"_allresnets_alllayers_mds.pdf")

    plt.figure(figsize=(17,7), dpi=400)

    blocks = dist.reshape((m, n, m, n)).transpose(0,2,1,3).reshape(m*m,n,n)
    all_idx = np.triu(np.arange(m*m).reshape(m,m),k=1)
    idx = all_idx[np.where(all_idx!=0)]

    model_names = ["1 2", "1 3", "1 4", "1 5", "2 3", "2 4", "2 5", "3 4", "3 5", "4 5"]

    model_idx = 0
    color_idx = 0
    
    cmap = plt.cm.get_cmap('viridis', 10)
    colors_six = ['red','orange','yellow','green','blue','darkviolet']

    model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model_layers, _ = get_graph_node_names(model5)

    stack_diags = np.zeros((int(m*(m-1)/2), n))

    fifth_model_idx = [4,9,14,19]
    
    for i in idx:
        diags = np.diag(blocks[i])
        stack_diags[model_idx, :] = diags
        
        if i in fifth_model_idx:
            #col = 'red'
            model_idx += 1
            continue
        else:
            #col='black'
            plt.plot(range(n), diags, '.-', linewidth=1, markersize=5, label="Model "+model_names[model_idx][0]+" vs Model "+model_names[model_idx][2], color=colors_six[color_idx])
            color_idx += 1 
            
        #plt.plot(range(n), diags, '.-', linewidth=1, markersize=5, label="Model "+model_names[model_idx][0]+" vs Model "+model_names[model_idx][2], color=col)
        
        #plt.plot(range(n), diags, '.-', linewidth=1, markersize=5, label="Model "+model_names[model_idx][0]+" vs Model "+model_names[model_idx][2], color=cmap(model_idx))
        model_idx += 1
    
    plt.xlabel("Layer")
    plt.xticks(ticks=range(n), labels=model_layers, rotation=90, fontsize=6)
    plt.grid(axis='x', color='gray', alpha=0.3, linewidth=0.5)
    plt.ylabel(dist_name.capitalize()+" Distance")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("figures/resnet50_results/all_layers/compare_model5_dist_"+dist_name+"_allresnets_alllayers_retrieval.svg", dpi=400)
    plt.savefig("figures/resnet50_results/all_layers/compare_without_model5_dist_"+dist_name+"_allresnets_alllayers_retrieval.svg", dpi=400)

    np.save('stacked_diags'+dist_name+'.npy', stack_diags)
    
    mean = stack_diags.mean(axis=0)
    var = stack_diags.var(axis=0, ddof=1)
    std = np.sqrt(var)
    
    if dist_name == 'wasserstein':
        norm_mean = mean/np.max(mean)
        means.append(norm_mean)
        norm_std = std/np.max(std)
        stds.append(norm_std)
    
    else:
        means.append(mean)
        stds.append(std)

    plt.figure(figsize=(14, 6), dpi=400)

    plt.plot(range(n), mean, '-o', markersize=2, linewidth=1, color='black')
    plt.fill_between(range(n), mean-std, mean+std, alpha=0.15, color='gray')
    plt.xticks(ticks=range(n), labels=model_layers, rotation=90, fontsize=6)
    plt.grid(axis='x', color='gray', alpha=0.1, linewidth=0.5)
    plt.xlabel("Layer")
    plt.ylabel(dist_name.capitalize()+" Distance")
    plt.tight_layout()
    plt.savefig("figures/resnet50_results/all_layers/avg_dist_"+dist_name+"_allresnets_alllayers_retrieval.svg", dpi=400)
'''
# compare the averages
'''
plt.figure(figsize=(14, 6), dpi=400)

for j, (mean_arr, std_arr) in enumerate(zip(means, stds)):
    x = np.arange(len(mean_arr))
    plt.plot(x, mean_arr, '-o', markersize=2, linewidth=1,
             color=main_colors[j], label=dist_names[j], zorder=3 + j)
    plt.fill_between(x, mean_arr - std_arr, mean_arr + std_arr,
                     alpha=0.15, color=error_colors[j], zorder=1 + j)

xtick_count = len(model_layers)
plt.xticks(ticks=range(xtick_count), labels=model_layers, rotation=90, fontsize=6)
plt.grid(axis='x', color='gray', alpha=0.1, linewidth=0.5)
plt.xlabel("Layer")
plt.ylabel("Distance")
plt.legend()
plt.tight_layout()
plt.savefig("figures/resnet50_results/all_layers/avg_dist_comparison_allresnets_alllayers_retrieval.svg", dpi=400)
'''
## conv vs bn vs relu
'''
import seaborn as sns
import pandas as pd

stacks_wass = np.load('stacked_diagswasserstein.npy')
stacks_jsd = np.load('stacked_diagsJSD.npy')

model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model_layers, _ = get_graph_node_names(model5)

conv_idx = []
bn_idx = []
relu_idx = []
for i, layer_name in enumerate(model_layers):
    if 'conv' in layer_name:
        conv_idx.append(i)
    elif 'bn' in layer_name:
        bn_idx.append(i)
    elif 'relu' in layer_name:
        relu_idx.append(i)

conv_wass = stacks_wass[:, conv_idx]
bn_wass = stacks_wass[:, bn_idx]
relu_wass = stacks_wass[:, relu_idx]

conv_jsd = stacks_jsd[:, conv_idx]
bn_jsd = stacks_jsd[:, bn_idx]
relu_jsd = stacks_jsd[:, relu_idx]

def to_rows(arr, measure, layer):
    return [(measure, layer, float(v)) for v in np.asarray(arr).ravel()]
 
rows = []
rows += to_rows(conv_wass, "Wasserstein", "conv")
rows += to_rows(bn_wass, "Wasserstein", "bn")
rows += to_rows(relu_wass, "Wasserstein", "relu")
rows += to_rows(conv_jsd, "JSD", "conv")
rows += to_rows(bn_jsd, "JSD", "bn")
rows += to_rows(relu_jsd, "JSD", "relu")
 
df = pd.DataFrame(rows, columns=["measure", "layer", "value"])
df["x"] = df["measure"] + "_" + df["layer"]
 
order = ["Wasserstein_conv", "Wasserstein_bn", "Wasserstein_relu", "JSD_conv", "JSD_bn", "JSD_relu"]
df["x"] = pd.Categorical(df["x"], categories=order, ordered=True)

# palette: warm for A, cool for B
pal = sns.color_palette("Reds", 3) + sns.color_palette("Blues", 3)

cat_order = ["conv", "bn", "relu"]

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

for ax, meas, palette in zip(
    axes,
    ["Wasserstein", "JSD"],
    ["Reds", "Blues"]
):
    d = df[df["measure"] == meas].copy()
 
    # make the x a *fresh* categorical with only the 3 cats
    d["layer"] = pd.Categorical(d["layer"], categories=cat_order, ordered=True)
 
    # plot (choose one: swarm only, or violin+swarm overlay)
    # --- violin behind (optional) ---
    # sns.violinplot(data=d, x="layer", y="value", order=cat_order,
    #                inner="quartile", cut=0, linewidth=1, saturation=0.9,
    #                color=sns.color_palette(palette, 4)[2], ax=ax)
    # --- swarm points ---
    sns.swarmplot(data=d, x="layer", y="value", order=cat_order,
                  palette=palette, size=4, ax=ax)
    sns.violinplot(data=d, x="layer", y="value", order=cat_order,
    palette=palette, inner=None, linewidth=0, cut=0, ax=ax, alpha=0.25)
    
    means = (
        d.groupby("layer")["value"].mean()
         .reindex(cat_order)          # ensure cat1, cat2, cat3 order
    )
 
    ax.scatter(
        means.index, means.values,
        s=120, marker="o",            # big diamond marker
        facecolor="gray", edgecolor="black", linewidth=1.6,
        zorder=10, label="Mean"
    )
 
    # clean axis labels
    ax.set_xlabel("")                     # only show cat1/cat2/cat3
    ax.set_ylabel("Distance")
    ax.set_xticklabels(cat_order)
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)
 
    # add group label UNDER the ticks
    fig.subplots_adjust(bottom=0.22)
    ax.text(0.5, -0.03, meas, transform=ax.transAxes,
            ha="center", va="top", fontsize=12)
 
plt.tight_layout()
plt.savefig("figures/resnet50_results/all_layers/wasserstein_jsd_conv_bn_relu_corrected_comparison.svg", dpi=400)
'''

# conv vs bn vs relu vs downsample vs avgpool vs fc
'''
import seaborn as sns
import pandas as pd

stacks_wass = np.load('stacked_diagswasserstein.npy')
stacks_jsd = np.load('stacked_diagsJSD.npy')

model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model_layers, _ = get_graph_node_names(model5)

conv_idx = []
bn_idx = []
relu_idx = []
downsample_idx = []
maxpool_idx = []
avgpool_idx = []
fc_idx = []

for i, layer_name in enumerate(model_layers):
    if 'conv' in layer_name:
        conv_idx.append(i)
    elif 'bn' in layer_name:
        bn_idx.append(i)
    elif 'relu' in layer_name:
        relu_idx.append(i)
    elif 'downsample' in layer_name:
        downsample_idx.append(i)
    elif 'avgpool' in layer_name:
        avgpool_idx.append(i)
    elif 'fc' in layer_name:
        fc_idx.append(i)

conv_wass = stacks_wass[:, conv_idx]
bn_wass = stacks_wass[:, bn_idx]
relu_wass = stacks_wass[:, relu_idx]
downsample_wass = stacks_wass[:, downsample_idx]
avgpool_wass = stacks_wass[:, avgpool_idx]
fc_wass = stacks_wass[:, fc_idx]

conv_jsd = stacks_jsd[:, conv_idx]
bn_jsd = stacks_jsd[:, bn_idx]
relu_jsd = stacks_jsd[:, relu_idx]
downsample_jsd = stacks_jsd[:, downsample_idx]
avgpool_jsd = stacks_jsd[:, avgpool_idx]
fc_jsd = stacks_jsd[:, fc_idx]

def to_rows(arr, measure, layer):
    return [(measure, layer, float(v)) for v in np.asarray(arr).ravel()]
 
rows = []
rows += to_rows(conv_wass, "Wasserstein", "conv")
rows += to_rows(bn_wass, "Wasserstein", "bn")
rows += to_rows(relu_wass, "Wasserstein", "relu")
rows += to_rows(downsample_wass, "Wasserstein", "downsample")
rows += to_rows(avgpool_wass, "Wasserstein", "avgpool")
rows += to_rows(fc_wass, "Wasserstein", "fc")

rows += to_rows(conv_jsd, "JSD", "conv")
rows += to_rows(bn_jsd, "JSD", "bn")
rows += to_rows(relu_jsd, "JSD", "relu")
rows += to_rows(downsample_jsd, "JSD", "downsample")
rows += to_rows(avgpool_jsd, "JSD", "avgpool")
rows += to_rows(fc_jsd, "JSD", "fc")
 
df = pd.DataFrame(rows, columns=["measure", "layer", "value"])
df["x"] = df["measure"] + "_" + df["layer"]
 
order = ["Wasserstein_conv", "Wasserstein_bn", "Wasserstein_relu", "Wasserstein_downsample", "Wasserstein_avgpool", "Wasserstein_fc", 
         "JSD_conv", "JSD_bn", "JSD_relu", "JSD_downsample", "JSD_avgpool", "JSD_fc"]

df["x"] = pd.Categorical(df["x"], categories=order, ordered=True)

cat_order = ["conv", "bn", "relu", "downsample", "avgpool", "fc"]

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

for ax, meas, palette in zip(
    axes,
    ["Wasserstein", "JSD"],
    ["Red", "Blue"]
):
    d = df[df["measure"] == meas].copy()
 
    d["layer"] = pd.Categorical(d["layer"], categories=cat_order, ordered=True)
 
    # --- swarm points ---
    sns.swarmplot(data=d, x="layer", y="value", order=cat_order,
                  palette=palette, size=4, ax=ax)
    sns.violinplot(data=d, x="layer", y="value", order=cat_order,
    palette=palette, inner=None, linewidth=0, cut=0, ax=ax, alpha=0.25)
    
    means = (
        d.groupby("layer")["value"].mean()
         .reindex(cat_order)          
    )
 
    ax.scatter(
        means.index, means.values,
        s=120, marker="o",       
        facecolor="gray", edgecolor="black", linewidth=1.6,
        zorder=10, label="Mean"
    )
 
    # clean axis labels
    ax.set_xlabel("")                   
    ax.set_ylabel("Distance")
    ax.set_xticklabels(cat_order)
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)
 
    # add group label UNDER the ticks
    fig.subplots_adjust(bottom=0.22)
    ax.text(0.5, -0.03, meas, transform=ax.transAxes,
            ha="center", va="top", fontsize=12)
 
plt.tight_layout()
plt.savefig("figures/resnet50_results/all_layers/wasserstein_jsd_allayers_comparison.svg", dpi=400)
'''
# Super detailed layers comparison

'''import seaborn as sns
import pandas as pd

stacks_wass = np.load('stacked_diagswasserstein.npy')
stacks_jsd = np.load('stacked_diagsJSD.npy')

model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model_layers, _ = get_graph_node_names(model5)

conv1_idx = []
conv2_idx = []
conv3_idx = []
bn1_idx = []
bn2_idx = []
bn3_idx = []
relu_idx = []
relu1_idx = []
relu2_idx = []
add_idx = []
downsample1_idx = []
downsample2_idx = []
maxpool_idx = []
avgpool_idx = []
fc_idx = []

for i, layer_name in enumerate(model_layers):
    if 'conv1' in layer_name:
        conv1_idx.append(i)
    elif 'conv2' in layer_name:
        conv2_idx.append(i)
    elif 'conv3' in layer_name:
        conv3_idx.append(i)
    elif 'bn1' in layer_name:
        bn1_idx.append(i)
    elif 'bn2' in layer_name:
        bn2_idx.append(i)
    elif 'bn3' in layer_name:
        bn3_idx.append(i)
    elif 'relu_1' in layer_name:
        relu1_idx.append(i)
    elif 'relu_2' in layer_name:
        relu2_idx.append(i)
    elif 'relu' in layer_name:
        relu_idx.append(i)
    elif 'add' in layer_name:
        add_idx.append(i)
    elif 'downsample.0' in layer_name:
        downsample1_idx.append(i)
    elif 'downsample.1' in layer_name:
        downsample2_idx.append(i)
    elif 'avgpool' in layer_name:
        avgpool_idx.append(i)
    elif 'maxpool' in layer_name:
        maxpool_idx.append(i)
    elif 'fc' in layer_name:
        fc_idx.append(i)

conv1_wass = stacks_wass[:, conv1_idx]
conv2_wass = stacks_wass[:, conv2_idx]
conv3_wass = stacks_wass[:, conv3_idx]
bn1_wass = stacks_wass[:, bn1_idx]
bn2_wass = stacks_wass[:, bn2_idx]
bn3_wass = stacks_wass[:, bn3_idx]
relu_wass = stacks_wass[:, relu_idx]
relu1_wass = stacks_wass[:, relu1_idx]
relu2_wass = stacks_wass[:, relu2_idx]
add_wass = stacks_wass[:, add_idx]
downsample1_wass = stacks_wass[:, downsample1_idx]
downsample2_wass = stacks_wass[:, downsample2_idx]
avgpool_wass = stacks_wass[:, avgpool_idx]
maxpool_wass = stacks_wass[:, maxpool_idx]
fc_wass = stacks_wass[:, fc_idx]

conv1_jsd = stacks_jsd[:, conv1_idx]
conv2_jsd = stacks_jsd[:, conv2_idx]
conv3_jsd = stacks_jsd[:, conv3_idx]
bn1_jsd = stacks_jsd[:, bn1_idx]
bn2_jsd = stacks_jsd[:, bn2_idx]
bn3_jsd = stacks_jsd[:, bn3_idx]
relu_jsd = stacks_jsd[:, relu_idx]
relu1_jsd = stacks_jsd[:, relu1_idx]
relu2_jsd = stacks_jsd[:, relu2_idx]
add_jsd = stacks_jsd[:, add_idx]
downsample1_jsd = stacks_jsd[:, downsample1_idx]
downsample2_jsd = stacks_jsd[:, downsample2_idx]
avgpool_jsd = stacks_jsd[:, avgpool_idx]
maxpool_jsd = stacks_jsd[:, maxpool_idx]
fc_jsd = stacks_jsd[:, fc_idx]

def to_rows(arr, measure, layer):
    return [(measure, layer, float(v)) for v in np.asarray(arr).ravel()]
 
rows = []
rows += to_rows(conv1_wass, "Wasserstein", "conv1")
rows += to_rows(conv2_wass, "Wasserstein", "conv2")
rows += to_rows(conv3_wass, "Wasserstein", "conv3")
rows += to_rows(bn1_wass, "Wasserstein", "bn1")
rows += to_rows(bn2_wass, "Wasserstein", "bn2")
rows += to_rows(bn3_wass, "Wasserstein", "bn3")
rows += to_rows(relu_wass, "Wasserstein", "relu")
rows += to_rows(relu1_wass, "Wasserstein", "relu1")
rows += to_rows(relu2_wass, "Wasserstein", "relu2")
rows += to_rows(add_wass, "Wasserstein", "add")
rows += to_rows(downsample1_wass, "Wasserstein", "downsample1")
rows += to_rows(downsample2_wass, "Wasserstein", "downsample2")
rows += to_rows(avgpool_wass, "Wasserstein", "avgpool")
rows += to_rows(maxpool_wass, "Wasserstein", "maxpool")
rows += to_rows(fc_wass, "Wasserstein", "fc")

rows += to_rows(conv1_jsd, "JSD", "conv1")
rows += to_rows(conv2_jsd, "JSD", "conv2")
rows += to_rows(conv3_jsd, "JSD", "conv3")
rows += to_rows(bn1_jsd, "JSD", "bn1")
rows += to_rows(bn2_jsd, "JSD", "bn2")
rows += to_rows(bn3_jsd, "JSD", "bn3")
rows += to_rows(relu_jsd, "JSD", "relu")
rows += to_rows(relu1_jsd, "JSD", "relu1")
rows += to_rows(relu2_jsd, "JSD", "relu2")
rows += to_rows(add_jsd, "JSD", "add")
rows += to_rows(downsample1_jsd, "JSD", "downsample1")
rows += to_rows(downsample2_jsd, "JSD", "downsample2")
rows += to_rows(avgpool_jsd, "JSD", "avgpool")
rows += to_rows(maxpool_jsd, "JSD", "maxpool")
rows += to_rows(fc_jsd, "JSD", "fc")
 
df = pd.DataFrame(rows, columns=["measure", "layer", "value"])
df["x"] = df["measure"] + "_" + df["layer"]
 
order = ["Wasserstein_conv1", "Wasserstein_conv2", "Wasserstein_conv3", 
         "Wasserstein_bn1", "Wasserstein_bn2", "Wasserstein_bn3",
         "Wasserstein_relu", "Wasserstein_relu1", "Wasserstein_relu2",
         "Wasserstein_add", "Wasserstein_downsample1", "Wasserstein_downsample2",
         "Wasserstein_avgpool", "Wasserstein_maxpool", "Wasserstein_fc", 
         "JSD_conv1", "JSD_conv2", "JSD_conv3", 
         "JSD_bn1", "JSD_bn2", "JSD_bn3",
         "JSD_relu", "JSD_relu1", "JSD_relu2",
         "JSD_add", "JSD_downsample1", "JSD_downsample2",
         "JSD_avgpool", "JSD_maxpool", "JSD_fc"]

df["x"] = pd.Categorical(df["x"], categories=order, ordered=True)

# cat_order = ["conv1", "conv2", "conv3",
#              "bn1", "bn2", "bn3",
#              "relu", "relu1", "relu2",
#              "add", "downsample1", "downsample2",
#              "avgpool", "maxpool", "fc"]

cat_order = [["maxpool", "relu2", "relu", "downsample1",
             "bn1", "downsample2", "conv1", "add",
             "conv2", "bn2", "bn3", "relu1", "conv3", 
             "avgpool", "fc"],
             ["maxpool", "relu2", "downsample1", "downsample2",
              "relu", "bn1", "add", "conv1", "relu1",
               "conv2", "bn2", "conv3", "bn3",   
               "avgpool", "fc"]]

fig, axes = plt.subplots(1, 2, figsize=(28, 10))

for ax, meas, palette, id in zip(
    axes,
    ["Wasserstein", "JSD"],
    ["red", "blue"], [0, 1]
):
    d = df[df["measure"] == meas].copy()
 
    d["layer"] = pd.Categorical(d["layer"], categories=cat_order[id], ordered=True)
 
    # --- swarm points ---
    sns.swarmplot(data=d, x="layer", y="value", order=cat_order[id],
                  color=palette, size=4, ax=ax)
    sns.violinplot(data=d, x="layer", y="value", order=cat_order[id],
    color=palette, inner=None, linewidth=0, cut=0, ax=ax, alpha=0.25)
    
    means = (
        d.groupby("layer")["value"].mean()
         .reindex(cat_order[id])          
    )
 
    ax.scatter(
        means.index, means.values,
        s=120, marker="o",       
        facecolor="gray", edgecolor="black", linewidth=1.6,
        zorder=10, label="Mean"
    )
 
    # clean axis labels
    ax.set_xlabel("")                   
    ax.set_ylabel("Distance")
    ax.set_xticklabels(cat_order[id])
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)
 
    # add group label UNDER the ticks
    fig.subplots_adjust(bottom=0.22)
    ax.text(0.5, -0.03, meas, transform=ax.transAxes,
            ha="center", va="top", fontsize=12)
 
plt.tight_layout()
plt.savefig("figures/resnet50_results/all_layers/wasserstein_jsd_allayers_comparison_detailed.svg", dpi=400)'''

### 

'''import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch

dist_jsd = np.load("dist_JSD_1000_all_resnets_all_layers.npy")
dist_wass = np.load("dist_wasserstein_1000_all_resnets_all_layers.npy")

n = int(len(dist_jsd)/5) # number of layers per model
m = 5 # number of models

model_pair_names = ["1 2", "1 3", "1 4", "1 5", "2 3", "2 4", "2 5", "3 4", "3 5", "4 5"]

blocks_jsd = dist_jsd.reshape((m, n, m, n)).transpose(0,2,1,3).reshape(m*m,n,n)
all_idx = np.triu(np.arange(m*m).reshape(m,m),k=1)
idx = all_idx[np.where(all_idx!=0)]

stack_diags_jsd = np.zeros((int(m*(m-1)/2), n))

model_idx = 0

for i in idx:
    diags = np.diag(blocks_jsd[i])
    stack_diags_jsd[model_idx, :] = diags
    model_idx +=1

blocks_wass = dist_wass.reshape((m, n, m, n)).transpose(0,2,1,3).reshape(m*m,n,n)
all_idx = np.triu(np.arange(m*m).reshape(m,m),k=1)
idx = all_idx[np.where(all_idx!=0)]

stack_diags_wass = np.zeros((int(m*(m-1)/2), n))

model_idx = 0

for i in idx:
    diags = np.diag(blocks_wass[i])
    stack_diags_wass[model_idx, :] = diags
    model_idx +=1

G, S = stack_diags_jsd.shape  # G=10, S=176
x = stack_diags_jsd.reshape(-1)
y = stack_diags_wass.reshape(-1)

group_idx = np.repeat(np.arange(G), S)
layer_idx   = np.tile(np.arange(S), G) 

# Model Pair Comparison Color Coding

cmap = plt.get_cmap('tab10')   # good discrete colormap
colors = [cmap(i % 10) for i in range(G)]

plt.figure(figsize=(12,10))
for g in range(G):
    mask = group_idx == g
    plt.scatter(x[mask], y[mask], color=colors[g], s=20, alpha=0.9, label="Model "+model_pair_names[g][0]+" vs Model "+model_pair_names[g][2])
 
plt.xlabel('JSD')
plt.ylabel('Wasserstein')
plt.title('Color Coded by Model Pair Comparison')
plt.legend(title='Model Pairs', frameon=True, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig("figures/resnet50_results/all_layers/jsd_vs_wass_model_pair.svg", dpi=400)

# Variations in Training Color Coding

colors = ['b', 'g', 'r', 'y']

var_names = ["seed", "seed+aug", "seed+aug+LR", "pretrained vs home trained"]

plt.figure(figsize=(12,10))
for g in range(G):
    if g == 0:
        label = var_names[0]
        col = colors[0]
    elif g == 7:
        label = var_names[1]
        col = colors[1]
    elif g in [1, 2, 4, 5]:
        if g ==1:
            label = var_names[2]
        else:
            label= None
        col = colors[2]
    elif g in [3, 6, 8, 9]:
        if g==3:
            label = var_names[3]
        else:
            label= None
        col = colors[3]
        
    mask = group_idx == g
    plt.scatter(x[mask], y[mask], color=col, s=20, alpha=0.9, label=label)
 
plt.xlabel('JSD')
plt.ylabel('Wasserstein')
plt.title('Color Coded by Variation Types')
plt.legend(title='Variation Types', frameon=True, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig("figures/resnet50_results/all_layers/jsd_vs_wass_variation_type.svg", dpi=400)

# Variations in Training Color Coding without 5th model comparisons

colors = ['b', 'g', 'r', 'y']

var_names = ["seed", "seed+aug", "seed+aug+LR", "pretrained vs home trained"]

plt.figure(figsize=(12,10))
for g in range(G):
    if g == 0:
        label = var_names[0]
        col = colors[0]
        mask = group_idx == g
        plt.scatter(x[mask], y[mask], color=col, s=20, alpha=0.9, label=label)
    elif g == 7:
        label = var_names[1]
        col = colors[1]
        mask = group_idx == g
        plt.scatter(x[mask], y[mask], color=col, s=20, alpha=0.9, label=label)
    elif g in [1, 2, 4, 5]:
        if g ==1:
            label = var_names[2]
        else:
            label= None
        col = colors[2]
        mask = group_idx == g
        plt.scatter(x[mask], y[mask], color=col, s=20, alpha=0.9, label=label)
    elif g in [3, 6, 8, 9]:
        if g==3:
            label = var_names[3]
        else:
            label= None
        col = colors[3]
        
plt.xlabel('JSD')
plt.ylabel('Wasserstein')
plt.title('Color Coded by Variation Types')
plt.legend(title='Variation Types', frameon=True, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig("figures/resnet50_results/all_layers/jsd_vs_wass_variation_type_without_5th.svg", dpi=400)

# Layer Depth Comparison Color Gradient

plt.figure(figsize=(14,10))
norm_layer = Normalize(vmin=np.min(layer_idx), vmax=np.max(layer_idx))
sc = plt.scatter(x, y, c=layer_idx, s=20, alpha=0.9, cmap='plasma', norm=norm_layer)
plt.xlabel('JSD')
plt.ylabel('Wasserstein')
plt.title('Color Gradient by Layer Depth')
cbar = plt.colorbar(sc)
cbar.set_label('Layer Depth')
plt.tight_layout()
#plt.show()
plt.savefig("figures/resnet50_results/all_layers/jsd_vs_wass_layer_depth.svg", dpi=400)'''


## Layer Name Color Coded
'''stack_diags_wass = np.load('stacked_diagswasserstein.npy')
stack_diags_jsd = np.load('stacked_diagsJSD.npy')

model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model_layers, _ = get_graph_node_names(model5)

layers = ['conv', 'bn', 'relu', 'downsample', 'add', 'avgpool', 'fc', 'other']

G, S = stack_diags_jsd.shape  # G=10, S=176
x = stack_diags_jsd.reshape(-1)
y = stack_diags_wass.reshape(-1)

layer_groups = []

for layer_name in model_layers:
    if 'conv' in layer_name:
        layer_groups.append(0)
    elif 'bn' in layer_name:
        layer_groups.append(1)
    elif 'relu' in layer_name:
        layer_groups.append(2)
    elif 'downsample' in layer_name:
        layer_groups.append(3)
    elif 'add' in layer_name:
        layer_groups.append(4)
    elif 'avgpool' in layer_name:
        layer_groups.append(5)
    elif 'fc' in layer_name:
        layer_groups.append(6)
    else:
        layer_groups.append(7)


group_idx = np.repeat(np.arange(G), S)
layer_idx   = np.tile(np.arange(S), G) 
layer_groups_idx = np.tile(layer_groups, G)

cmap = plt.get_cmap('tab10')   # good discrete colormap
colors = [cmap(i % 8) for i in range(8)]

plt.figure(figsize=(14,10))
for j in range(8):
    label = layers[j]
    col = colors[j]
    mask = layer_groups_idx == j
    plt.scatter(x[mask], y[mask], color=col, s=20, alpha=0.9, label=label)


plt.xlabel('JSD')
plt.ylabel('Wasserstein')
plt.title('Color Coded by Layer Name')
plt.legend(title='Model Pairs', frameon=True, loc='best')
plt.tight_layout()
#plt.show()
plt.savefig("figures/resnet50_results/all_layers/jsd_vs_wass_layer_name.svg", dpi=400)
'''


 ## Avergae/ranked block matrix plot
 
'''from scipy.stats import rankdata

dist_names = ['wasserstein', 'JSD']

for dist_name in dist_names:
    
    dist = np.load("dist_"+dist_name+"_1000_all_resnets_all_layers.npy")

    n = int(len(dist)/5) # number of layers per model
    m = 5 # number of models

    blocks = dist.reshape((m, n, m, n)).transpose(0,2,1,3).reshape(m*m,n,n)
    all_idx = np.triu(np.arange(m*m).reshape(m,m),k=1)
    idx = all_idx[np.where(all_idx!=0)]

    model_names = ["1 2", "1 3", "1 4", "1 5", "2 3", "2 4", "2 5", "3 4", "3 5", "4 5"]

    model_idx = 0
    color_idx = 0
    
    cmap = plt.cm.get_cmap('viridis', 10)
    colors_six = ['red','orange','yellow','green','blue','darkviolet']

    model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model_layers, _ = get_graph_node_names(model5)
    
    stack_blocks = []
    
    for i in idx:
        stack_blocks.append(blocks[i])
    
    np_stack_blocks=np.array(stack_blocks)
    avg_stack_blocks = np.mean(np_stack_blocks, axis=0)
    
    ranked_blocks = rankdata(np.reshape(avg_stack_blocks, (np_stack_blocks.shape[1]*np_stack_blocks.shape[1], 1)), 'min').reshape(np_stack_blocks.shape[1], np_stack_blocks.shape[1])
    
    # plt.figure()
    # plt.imshow(avg_stack_blocks, "bone", vmin=0, vmax=np.max(avg_stack_blocks))
    # plt.colorbar()
    # plt.title(dist_name.capitalize()+" Distance")
    # ax = plt.gca()
    # ax.set_axis_off()
    # plt.savefig("figures/resnet50_results/all_layers/avged_blocks_dist_"+dist_name+"_allresnets_alllayers.svg", dpi=800)
    
    plt.figure()
    plt.imshow(ranked_blocks, "bone", vmin=0, vmax=np.max(ranked_blocks))
    plt.colorbar()
    plt.title(dist_name.capitalize()+" Distance")
    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig("figures/resnet50_results/all_layers/ranked_blocks_dist_"+dist_name+"_allresnets_alllayers.svg", dpi=800)'''

## Block and layer specific retrieval plot

'''dist_names = ['wasserstein', 'JSD']

model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model_layers, _ = get_graph_node_names(model5)

for dist_name in dist_names:
    
    stack_diags = np.load('stacked_diags'+dist_name+'.npy')
    
    fifth_model_idx = [3, 6, 8, 9]
    all_others = [0, 1, 2, 4, 5, 7]
    
    # uncomment this for the analysis without the 5th model
    #stack_diags = stack_diags[all_others, :]
    
    conv_idx = []
    bn_idx = []
    relu_idx = []
    downsample_idx = []
    maxpool_idx = []
    avgpool_idx = []
    fc_idx = []

    for i, layer_name in enumerate(model_layers):
        if 'conv' in layer_name:
            conv_idx.append(i)
        elif 'bn' in layer_name:
            bn_idx.append(i)
        elif 'relu' in layer_name:
            relu_idx.append(i)
        elif 'downsample' in layer_name:
            downsample_idx.append(i)
        elif 'avgpool' in layer_name:
            avgpool_idx.append(i)
        elif 'fc' in layer_name:
            fc_idx.append(i)

    layers = [conv_idx, bn_idx, relu_idx]
    
    colors = ['red', 'blue', 'green']
    colors_std = ['mistyrose', 'lightsteelblue','lightgreen']
    
    layer_labels = ['conv', 'bn', 'relu']
    
    x_tick_labels = []
    x_tick_colors = []
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=400)
    
    for i, layer in enumerate(layers):
        
        layer_stack = stack_diags[:, layer]
        mean = layer_stack.mean(axis=0)
        var = layer_stack.var(axis=0, ddof=1)
        std = np.sqrt(var)
        
        n = len(mean)
        
        bottleneck_no = 0
        
        plt.plot(range(n), mean, '-o', markersize=2, linewidth=1, color=colors[i], label=layer_labels[i])
        plt.fill_between(range(n), mean-std, mean+std, alpha=0.15, color=colors_std[i])
        if i ==0:
            for j in layer:
                if j != 1:
                    x_tick_labels.append(model_layers[j][:8]+"."+str((bottleneck_no%3)+1))
                    
                    if (bottleneck_no%3)+1 == 3:
                        x_tick_colors.append('red')
                    else:
                        x_tick_colors.append('black')
                        
                    bottleneck_no += 1
                    
                else:
                    x_tick_labels.append('stem')
        
    plt.xticks(ticks=range(n), labels=x_tick_labels, rotation=90, fontsize=6)
    for t in ax.get_xticklabels():
        txt = t.get_text()
        if txt[-1] == '3':
            t.set_color('red')
        else:
            continue
    
    plt.xlabel("Layer")
    plt.ylabel(dist_name.capitalize()+" Distance")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='x', color='gray', alpha=0.1, linewidth=0.5)
    #plt.show()
    #plt.savefig("figures/resnet50_results/all_layers/avg_dist_"+dist_name+"_per_layer_type_retrieval_without5th.svg", dpi=400)
    plt.savefig("figures/resnet50_results/all_layers/avg_dist_"+dist_name+"_per_layer_type_retrieval.svg", dpi=400)'''
        

## Relu-2 & downsample specific retrieval plot

'''dist_names = ['wasserstein', 'JSD']

model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model_layers, _ = get_graph_node_names(model5)

for dist_name in dist_names:
    
    stack_diags = np.load('stacked_diags'+dist_name+'.npy')
    
    fifth_model_idx = [3, 6, 8, 9]
    all_others = [0, 1, 2, 4, 5, 7]
    
    # uncomment this for the analysis without the 5th model
    #stack_diags = stack_diags[all_others, :]
    
    relu_idx = []
    downsample_idx = []

    for i, layer_name in enumerate(model_layers):

        if 'relu_2' in layer_name:
            relu_idx.append(i)
        elif 'downsample' in layer_name:
            downsample_idx.append(i)

    layers = [relu_idx, downsample_idx]
    
    colors = ['green', 'orange']
    colors_std = ['lightgreen', 'salmon']
    
    layer_labels = ['relu', 'downsample']
    
    for i, layer in enumerate(layers):
        
        x_tick_labels = []
        
        if i == 0:
            plt.figure(figsize=(12, 6), dpi=400)
            
        if i == 1:
            plt.figure(figsize=(10, 6), dpi=400)
        
        layer_stack = stack_diags[:, layer]
        mean = layer_stack.mean(axis=0)
        var = layer_stack.var(axis=0, ddof=1)
        std = np.sqrt(var)
        
        n = len(mean)
        
        bottleneck_no = 0
        
        #plt.errorbar(range(n), mean, yerr=mean+std, fmt='o', color=colors[i], label=layer_labels[i], capsize=5, markersize=6, linestyle='none')
        plt.plot(range(n), mean, '-o', markersize=2, linewidth=1, color=colors[i], label=layer_labels[i])
        plt.fill_between(range(n), mean-std, mean+std, alpha=0.15, color=colors_std[i])
        
        if i == 0:
            for j in layer:
                if j != 1:
                    x_tick_labels.append(model_layers[j][:8]+"."+str((bottleneck_no%3)+1))
                    bottleneck_no += 1
                    
                else:
                    x_tick_labels.append('stem')
        
            plt.xticks(ticks=range(n), labels=x_tick_labels, rotation=90, fontsize=6)
        
        if i == 1:
            for j in layer:
                if j != 1:
                    x_tick_labels.append(model_layers[j][:8]+"."+str((bottleneck_no%2)+1))
                    bottleneck_no += 1
                    
                else:
                    x_tick_labels.append('stem')
        
            plt.xticks(ticks=range(n), labels=x_tick_labels, rotation=90, fontsize=6)
        
        plt.title(layer_labels[i].capitalize())
        plt.xlabel("Layer")
        plt.ylabel(dist_name.capitalize()+" Distance")
        #plt.legend()
        plt.tight_layout()
        plt.grid(axis='x', color='gray', alpha=0.1, linewidth=0.5)
        #plt.show()
        plt.savefig("figures/resnet50_results/all_layers/"+layer_labels[i]+"_avg_dist_"+dist_name+"_per_layer_type_retrieval.svg", dpi=400)'''
        
## Relu_2 vs Add
'''
dist_names = ['wasserstein', 'JSD']

model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model_layers, _ = get_graph_node_names(model5)

fifth_model_idx = [3, 6, 8, 9]
all_others = [0, 1, 2, 4, 5, 7]

for dist_name in dist_names:
    
    stack_diags = np.load('stacked_diags'+dist_name+'.npy')
    
    relu_idx = []
    add_idx = []

    for i, layer_name in enumerate(model_layers):

        if 'relu_2' in layer_name:
            relu_idx.append(i)
        elif 'add' in layer_name:
            add_idx.append(i)

    relu_vals = stack_diags[:, relu_idx]
    add_vals = stack_diags[:, add_idx]
    
    plt.figure(figsize=(12,10))
    
    for i in range (relu_vals.shape[0]):
        sc = plt.scatter(add_vals[i,:], relu_vals[i,:], c=relu_idx, s=20, alpha=0.9, cmap='plasma')
    # for i in all_others:
    #     sc = plt.scatter(add_vals[i,:], relu_vals[i,:], c=relu_idx, s=20, alpha=0.9, cmap='plasma')
    
    
    plt.xlabel('Add')
    plt.ylabel('ReLu_2')
    
    if dist_name =='wasserstein':
        plt.xlim(0,3.2)
        plt.ylim(0, 3.2)
        plt.xticks(np.arange(0.0, 3.4, 0.2))
        plt.yticks(np.arange(0.0, 3.4, 0.2))
        plt.plot([0, 3.2], [0, 3.2], linestyle='--', color='gray', linewidth=1)
    elif dist_name =='JSD':
        plt.xlim(0,0.85)
        plt.ylim(0,0.85)
        plt.xticks(np.arange(0.0, 0.85, 0.1))
        plt.yticks(np.arange(0.0, 0.85, 0.1))
        plt.plot([0, 0.85], [0, 0.85], linestyle='--', color='gray', linewidth=1)
        
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('ReLu_2 vs Add')
    cbar = plt.colorbar(sc)
    cbar.set_label('Layer Depth')
    plt.tight_layout()
    #plt.show()
    plt.savefig("figures/resnet50_results/all_layers/relu_vs_add_dist_"+dist_name+".svg", dpi=400)
    #plt.savefig("figures/resnet50_results/all_layers/relu_vs_add_dist_"+dist_name+"_without_5thmodel.svg", dpi=400)
'''

## Kornblith plots layer specific
'''
dist_names = ['wasserstein', 'JSD']

model5 = torchvision.models.get_model("resnet50", weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model_layers, _ = get_graph_node_names(model5)

colorbar_max_vals = [8, 1]

for d, dist_name in enumerate(dist_names):
    
    dist = np.load("dist_"+dist_name+"_1000_all_resnets_all_layers.npy")
    
    n = int(len(dist)/5) # number of layers per model
    m = 5 # number of models
    
    blocks = dist.reshape((m, n, m, n)).transpose(0,2,1,3).reshape(m*m,n,n)
    all_idx = np.triu(np.arange(m*m).reshape(m,m),k=1)
    idx = all_idx[np.where(all_idx!=0)]
    
    conv_idx = []
    bn_idx = []
    relu_idx = []
    downsample_idx = []
    maxpool_idx = []
    avgpool_idx = []
    fc_idx = []

    for i, layer_name in enumerate(model_layers):
        if 'conv' in layer_name:
            conv_idx.append(i)
        elif 'bn' in layer_name:
            bn_idx.append(i)
        elif 'relu' in layer_name:
            relu_idx.append(i)
        elif 'downsample' in layer_name:
            downsample_idx.append(i)
        elif 'avgpool' in layer_name:
            avgpool_idx.append(i)
        elif 'fc' in layer_name:
            fc_idx.append(i)

    layers = [conv_idx, bn_idx, relu_idx]
    
    layer_labels = ['conv', 'bn', 'relu']
    
    stack_blocks = []
        
    for i in idx:
        stack_blocks.append(blocks[i])
    
    np_stack_blocks=np.array(stack_blocks)
    avg_stack_blocks = np.mean(np_stack_blocks, axis=0)
    
    for i, layer in enumerate(layers):
        
        layer_specific_stack = []
        
        for p in stack_blocks:
            
            mtx_per_block = np.zeros((len(layer), len(layer)))
            
            idx_1 = 0            
            for k in layer:
                idx_2 = 0
                for l in layer:
                    mtx_per_block[idx_1, idx_2] = p[k,l]
                    mtx_per_block[idx_2, idx_1] = p[k,l]
                    idx_2 += 1
                idx_1 += 1
                
            layer_specific_stack.append(mtx_per_block)
                
        np_layer_specific_stack=np.array(layer_specific_stack)
        avg_layer_specific = np.mean(np_layer_specific_stack, axis=0)
        
        plt.figure()
        plt.imshow(avg_layer_specific, "bone", vmin=0, vmax=np.max(colorbar_max_vals[d]))
        plt.colorbar()
        plt.title(layer_labels[i]+" Layers "+ dist_name.capitalize()+" Distance")
        ax = plt.gca()
        ax.set_axis_off()
        plt.savefig("figures/resnet50_results/all_layers/"+layer_labels[i]+"_blocks_dist_"+dist_name+"_allresnets_alllayers.svg", dpi=800)
'''       
