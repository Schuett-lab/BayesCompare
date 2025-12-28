import os

def get_dirs():

    cwd = os.getcwd()

    dirs = {}

    # paths for Meluxina
    if 'u103388' in cwd:
        dirs = {
        "checkpoint_main_path": "/project/home/p201045/outputs/",
        "input_images": "/project/home/p201045/mscoco/unlabeled2017.zip",
        "result_path": "/project/scratch/p201045/model-compare/results/",
        "figure_path": "/project/scratch/p201045/model-compare/figures/",
        "configs_path": "/home/users/u103388/BayesCompare/config_files/"}
    
    # paths for ULHPC
    elif 'soral' in cwd:
        dirs = {
        "checkpoint_main_path": "/home/users/soral/ModelTraining/outputs/",
        "input_images": "-",
        "result_path": "/home/users/soral/BayesCompare/results/",
        "figure_path": "/home/users/soral/BayesCompare/figures/",
        "configs_path": "/home/users/soral/BayesCompare/config_files/"}
        
    return dirs

