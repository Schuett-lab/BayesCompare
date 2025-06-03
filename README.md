# Bayesian Comparisons Between Representations
This code implements comparisons between predictive distributions as a metric for comparisons and is meant to acompany a submitted paper with the same title.

To run this code you will need to install the requirements and the package. Then you can run the scripts
in the scripts folder to reproduce the analyses in the paper.

## Install
You first need to install pytorch (and torchvision), which is not found if you just run `pip install .` or something similar. To do so follow the instructions [here](https://pytorch.org):
The command for conda without cuda is:
```
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

Then you can install the package using `pip install .`, which should also get all other requirements.

If you want to replicate our analyses you additionally need the MS COCO unlabeled image data. This data is avaliable here: [http://images.cocodataset.org/zips/unlabeled2017.zip](http://images.cocodataset.org/zips/unlabeled2017.zip). Unzip the folder and put it into a subfolder `images/unlabeled2017` or adjust im_folder in the `scripts/networks/py` and `scripts/example.py` to the path where the images are saved on your machine.

## Usage
The main comparison functions are implemented in src/BayesDist/normals.py . The other functions are just helper functions.
Functions that compute the Jensen-Shannon Divergence start with `jsd` and ones that compute the Total Variation Distance start with `tvd`. For each there is a version for general Gaussian functions and one that is just based on two covariance matrices. For the second type `eye_w` is called a in the paper. These functions are all you need for running your own analyses.

You find scripts to implement the analyses in the paper in the scripts folder:
- `illustrate.py` creates the elements for the illustration in Fig. 1.
- `example.py` creates the example analysis in Fig. 3.

For the other analyses you first need to run `networks.py` which runs the neural networks on 1000 test images and saves a file called `covs_1000.npy` with the covariance matrices for each network and each layer. After running this you can run the following two scripts:
- `chooseA.py` creates the plots in Fig. 2 illustrating the dependence between a and typical DNN distances.
- `Nimages.py` runs the analysis comparing the different metrics to each other and repeating each analysis with a sample of 25/50/100 images of the 1000 such that we can estimate the reliability of analyses. The results are in Fig. 4 in the paper.
- `numerics.py` runs the analysis for the number of samples presented in detail in the appendix Fig. 5.

Most of the scripts contain a point at which they save their computational results before loading and plotting them. 
