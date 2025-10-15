import torch
import torchvision
import torchlens as tl
import numpy as np
import os
import PIL
import BayesCompare
import tqdm
from torchvision import transforms as tvt
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names, modified_create_feature_extractor
import time

## First obtain the Covs matrix from activations

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

covs1 = []

model_num = 1

set_num = 1

def cov_funct(activations):
    cov = BayesCompare.get_cov(activations)
    return cov

# check if it is due to in place relu:
# yes it was indeed due to inplace relus. now modified feature extractor and feature extractor + covs results are the same 
def ReLU_inplace_to_False(module):
    for layer in module._modules.values():
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)

with torch.inference_mode():
    
    for model in tqdm.tqdm(models, position=0, desc="Models"):
        
        for set in tqdm.tqdm(sets, position=1, desc="Layer Sets"):
            
            ReLU_inplace_to_False(model)
            
            feature_extractor = create_feature_extractor(model, return_nodes=set)
            
            feats = feature_extractor(x_input)
            
            covs1.extend([BayesCompare.get_cov(feats[key]).detach().cpu().numpy()
                    for key in tqdm.tqdm(feats.keys(), position=2, desc="Cov Computation")])
        
            covs_extractor = modified_create_feature_extractor(model, return_nodes=set, cov_func=cov_funct)
            
            covs2 = covs_extractor(x_input)

            #torch.save(feats, "/home/sezan/Documents/BayesCompare/outputs/activations/feats_trial_model"+str(model_num)+"_set"+str(set_num)+".pth")
            
            set_num += 1
    
        model_num+=1

#np.save("covs_1000_all_resnets_densesampled.npy", np.stack(covs))

## Then using the Covs matrix, obtain the distances

# eye_w = 10/11

# covs = np.load("/home/sezan/Documents/BayesCompare/covs_1000_resnet50_densesampled.npy")

# dist = np.zeros((len(covs), len(covs)))
# for i, ci in tqdm.tqdm(enumerate(covs), total=len(covs)):
#     for j, cj in tqdm.tqdm(enumerate(covs), total=len(covs), position=1):
#         if j > i:
#             dist[i, j] = BayesCompare.jsd_normal_sig(ci, cj, 10000, eye_w=eye_w)
#             dist[j, i] = dist[i, j]
            
# np.save("dist_resnet_1000_densesampled.npy", dist)