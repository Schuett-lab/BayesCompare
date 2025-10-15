import torch
import torchvision
import torchlens as tl
import numpy as np
import os
import PIL
import BayesCompare
import tqdm
from torchvision import transforms as tvt
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import time
from collections import OrderedDict
from torchvision.models.feature_extraction import DualGraphModule
#from torchvision.transforms import v2 as tvt

al_select = [0, 1]

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
model3.load_state_dict(snapshot1["model"])
model4.load_state_dict(snapshot2["model"])

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

# what type of transform should I apply? 
'''
transforms = []
crop_size = 224,
resize_size = 232,
mean=(0.485, 0.456, 0.406),
std=(0.229, 0.224, 0.225),
interpolation=tvt.functional.InterpolationMode.BILINEAR,

transforms  = [tvt.PILToTensor(),
               tvt.Resize(resize_size, interpolation=interpolation, antialias=True), 
               tvt.CenterCrop(crop_size),
               tvt.ToDtype(torch.float, scale=True),
               tvt.Normalize(mean=mean, std=std)]

transforms = tvt.Compose(transforms)
'''
transform = tvt.Compose([tvt.Resize(256),
                         tvt.CenterCrop(224),
                         tvt.ToTensor(),
                         tvt.Normalize(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                         )
                        ])

transformed_ims = [transform(im.convert('RGB')) for im in ims]
x_input = torch.stack(transformed_ims)

# torchvision function

# check the layer names with train_nodes before getting the features
#train_nodes1, _ = get_graph_node_names(model1)
#train_nodes2, _ = get_graph_node_names(model2)
#train_nodes3, _ = get_graph_node_names(model3)

# model_layers = [
#     "conv1", "bn1", "relu", "maxpool",
#     "layer1.0.conv1", "layer1.0.bn1", "layer1.0.relu",
#     "layer1.0.conv3", "layer1.0.bn3", "layer1.0.downsample.0", "layer1.0.downsample.1", "layer1.0.add", "layer1.0.relu_2",
#     "layer1.1.conv3", "layer1.1.bn3", "layer1.1.add", "layer1.1.relu_2",
#     "layer1.2.conv1", "layer1.2.bn1", "layer1.2.relu",
#     "layer2.0.conv1", "layer2.0.bn1", "layer2.0.relu",
#     "layer2.0.conv3", "layer2.0.bn3", "layer2.0.downsample.0", "layer2.0.downsample.1", "layer2.0.add", "layer2.0.relu_2",
#     "layer2.2.conv1", "layer2.2.bn1", "layer2.2.relu",
#     "layer2.2.conv3", "layer2.2.bn3", "layer2.2.add", "layer2.2.relu_2",
#     "layer2.3.conv1", "layer2.3.bn1", "layer2.3.relu",
#     "layer2.3.conv3", "layer2.3.bn3", "layer2.3.add", "layer2.3.relu_2",
#     "layer3.0.conv3", "layer3.0.bn3", "layer3.0.downsample.0", "layer3.0.downsample.1", "layer3.0.add", "layer3.0.relu_2",
#     "layer3.1.conv2", "layer3.1.bn2", "layer3.1.relu_1", "layer3.1.conv3", "layer3.1.bn3", "layer3.1.add", "layer3.1.relu_2",
#     "layer3.2.conv3", "layer3.2.bn3", "layer3.2.add", "layer3.2.relu_2",
#     "layer3.3.conv1", "layer3.3.bn1", "layer3.3.relu",
#     "layer3.4.conv1", "layer3.4.bn1", "layer3.4.relu",
#     "layer3.4.conv2", "layer3.4.bn2", "layer3.4.relu_1",
#     "layer3.4.conv3", "layer3.4.bn3", "layer3.4.add", "layer3.4.relu_2",
#     "layer3.5.conv3", "layer3.5.bn3", "layer3.5.add", "layer3.5.relu_2",
#     "layer4.0.conv1", "layer4.0.bn1", "layer4.0.relu",
#     "layer4.0.conv3", "layer4.0.bn3", "layer4.0.downsample.0", "layer4.0.downsample.1", "layer4.0.add",
#     "layer4.2.conv3", "layer4.2.bn3", "layer4.2.add", "layer4.2.relu_2",
#     "avgpool", "fc"
# ]

# model_layers = ["conv1", "bn1","relu", "maxpool", "layer1.1.conv3", "layer1.2.conv3", "layer2.1.conv3", "layer2.2.conv3", "layer3.1.conv3", 
#                 "layer3.2.conv3", "layer4.1.conv3", "layer4.2.conv3", "avgpool","fc"]

#model_layers = ["layer1.2.conv3", "layer4.2.conv3", "avgpool"]

model_layers = ["layer1.2.conv3"]

covs = []

class CovFeatureExtractor(torch.fx.GraphModule):
    def __init__(self, root, graph, class_name, covs):
        super().__init__(root, graph, class_name)
        self.covs = covs

    def forward(self, x):
        activations = super().forward(x)
        for key, value in activations.items():
            self.covs.extend(BayesCompare.get_cov(value))
            del activations[key]
        # Optionally return something, e.g. self.covs or activations
        return self.covs
            
def cov_funct(activations):
    covs = [BayesCompare.get_cov(activations.detach().cpu().numpy())]
    
    return covs
    

with torch.inference_mode():
    
    for model in models:
        
        feature_extractor = create_feature_extractor(model, return_nodes=model_layers, cov_func=cov_funct)
        
        #cov_extractor = CovFeatureExtractor(model, feature_extractor.graph, feature_extractor.__class__.__name__, covs)
        #cov_extractor = CovFeatureExtractor(feature_extractor, feature_extractor.graph, feature_extractor.__class__.__name__, covs)
        #new_covs = cov_extractor(x_input)

        feats = feature_extractor(x_input)

        covs.extend([BayesCompare.get_cov(feats[key]).detach().cpu().numpy()
                    for key in tqdm.tqdm(feats.keys())])

np.save("covs_1000_resnet50_trial.npy", np.stack(covs))

# torchlens
'''
model_layers = ["conv2d_1_1", "batchnorm_1_2","relu_1_3", "maxpool2d_1_4", "conv2d_8_23", "conv2d_11_33", "conv2d_18_55", "conv2d_21_65", "conv2d_31_97", 
                "conv2d_34_107", "conv2d_50_159", "conv2d_53_169", "adaptiveavgpool2d_1_173","linear_1_175"]

responses = []

covs = []

start = time.time()

with torch.inference_mode():
    
    model_history1 = tl.log_forward_pass(model1, x_input, layers_to_save=model_layers, vis_opt="none")
    
    for layer in model_layers:
        responses.append(model_history1[layer].tensor_contents)

    for i in tqdm.tqdm(range(len(responses))):
        covs.extend([BayesCompare.get_cov(responses[i]).detach().cpu().numpy()])
 
end = time.time()
print(end - start)   
np.save("covs_1000_resnet50_torchlens.npy", np.stack(covs))

# ------


model_layers = ["conv2d_1_1", "batchnorm_1_2","relu_1_3", "maxpool2d_1_4", "conv2d_8_23", "conv2d_11_33", "conv2d_18_55", "conv2d_21_65", "conv2d_31_97", 
                "conv2d_34_107", "conv2d_50_159", "conv2d_53_169", "adaptiveavgpool2d_1_173","linear_1_175"]

covs = []

c = 1
start = time.time()
with torch.inference_mode():
    
    for model in models:
        
        model_history = tl.log_forward_pass(model, x_input, layers_to_save=model_layers, vis_opt="none")
        
        responses = []
        
        for layer in model_layers:
            responses.append(model_history[layer].tensor_contents)
        
        print("Computing the covs for model " + str(c))    
        covs.extend([BayesCompare.get_cov(responses[i]).detach().cpu().numpy()
                    for i in tqdm.tqdm(range(len(responses)))])
        
        c += 1
        
end = time.time()
print(end - start)'''
