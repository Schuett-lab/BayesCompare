import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

dist = np.load("dist_jsd_1000_resnet50_densesampled.npy")

a = np.arange(25).reshape(5,5)
b = np.ones((3,3))
c = np.kron(a, b)
d = c.reshape((5, 3, 5, 3)).transpose(0,2,1,3).reshape(25,3,3)

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
plt.savefig("figures/resnet50_results/dist_jsd_allresnets_densesampled_mds.svg")
plt.savefig("figures/resnet50_results/dist_jsd_allresnets_densesampled_mds.pdf")

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
plt.ylabel("JSD Distance")
plt.legend()
plt.tight_layout()
plt.savefig("figures/resnet50_results/dist_jsd_allresnets_densesampled_retrieval.svg", dpi=400)


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
plt.savefig("figures/resnet50_results/avg_dist_jsd_allresnets_densesampled_retrieval.svg", dpi=400)
