import numpy as np
import BayesCompare
import matplotlib.pyplot as plt
import os
import PIL
import BayesCompare
from torchvision import transforms as tvt
import torch
import torchvision
import tqdm
from sklearn.manifold import MDS

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

covs = np.load("/home/sezan/Documents/BayesCompare/covs_1000_all_resnets_densesampled.npy")

dist = BayesCompare.measure_dist(covs, checkpoint_dir="/home/sezan/Documents/BayesCompare/my_dist_checkpoints/", meas_name=['wasserstein'], alpha=10/11)

dist = dist['wasserstein']

np.save("dist_wasserstein_corrected_1000_all_resnets_densesampled", dist)


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
mds.fit(normalized_dists)

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
plt.ylabel("JSD Distance")
plt.tight_layout()
plt.savefig("figures/resnet50_results/avg_dist_wasserstein_corrected_corrected_allresnets_densesampled_retrieval.svg", dpi=400)

