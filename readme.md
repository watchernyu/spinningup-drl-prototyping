# Soft Actor-Critic Pytorch Implementation
Soft Actor-Critic Pytorch Implementation, based on the OpenAI Spinup documentation and some of its code base. This is a minimal, easy-to-learn and well-commented Pytorch implementation, and recommended to be studied along with the OpenAI Spinup Doc. This SAC implementation is based on the OpenAI spinningup repo, and uses spinup as a dependency. Target audience of this repo is Pytorch users (especially NYU students) who are learning Soft Actor-Critic algorithm. 

## Setup environment:
To use the code you should first download this repo, and then install this repo with the same method to install the original spinup repo. (So you are using the same method, but you should install this repo, not the original repo.)

the spinup documentation is here, you should read it to make sure you know the procedure: https://spinningup.openai.com/en/latest/user/installation.html

The only difference in installation is you want to install this repo, instead of the original repo, don't download the original repo, use this repo please. When you are ready to install this in a virtualenv (and don't forget to actually enter your virtualenv) you should first clone this repo onto your machine, enter the repo folder, and then use the pip install command (assuming your have a conda virtualenv with the name "rl"): 

Intall MuJoCo, gym, and other dependencies, and this repository: 

On Linux:
Go to a place on your machine where you can put python files. (for example, desktop or home, or create a folder), make sure you have Anaconda on your machine, then run the following commands in your terminal, which will create a conda environment called rl and then install for you:

```
conda create -n rl python=3.6
source activate rl 
conda install -y pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch
git clone https://github.com/openai/gym.git
cd gym
git checkout a4adef2
pip install -e .
cd ..  
git clone https://github.com/openai/mujoco-py
cd mujoco-py
git checkout 498b451
pip install -e . --no-cache
pip install -r requirements.txt
pip install -r requirements.dev.txt
cd ..
git clone https://github.com/watchernyu/spinningup-drl-prototyping.git
cd spinningup-drl-prototyping
pip install numpy==1.16.4
pip install tensorflow==1.12.0
pip install seaborn==0.8.1
pip install -e .
```

On OSX, replace the line that install pytorch with following: 
```
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```

For MuJoCo you will need to download the MuJoCo files for your system, and then have a license. 

Some of the above commands are not necessary if you are an expert. But they might help you if you are a beginner...

The Pytorch version used is: 1.2 (1.2-1.5 might all work), install pytorch:
https://pytorch.org/

If you want to run Mujoco environments, you need to also setup Mujoco. For how to install and run Mujoco on NYU's hpc cluster, check out my other tutorial: https://github.com/watchernyu/hpc_setup

## Run experiment
The SAC and SAC adaptive implementation can be found under `spinup/algos/sac_pytorch/`

Run experiments with pytorch sac: 

In the sac_pytorch folder, run the SAC code with `python sac_pytorch`

Note: currently there is no parallel running for SAC (also not supported by spinup), so you should always set number of cpu to 1 when you use experiment grid.

The program structure, though in Pytorch has been made to be as close to spinup tensorflow code as possible so readers who are familiar with other algorithm code in spinup will find this one easier to work with. I also referenced rlkit's SAC pytorch implementation, especially for the policy and value models part, but did a lot of simplification. 

Consult Spinup documentation for output and plotting:

https://spinningup.openai.com/en/latest/user/saving_and_loading.html

https://spinningup.openai.com/en/latest/user/plotting.html

Features of original spinup are mostly supported. In addition, we have some new features:

`sample_hpc_scripts` folder contains sample scripts that you can use to run parallel job arrays on the hpc. 

`sample_plot_helper` contains a short sample program that can help you do plotting in a more automatic way. The program can be tested to plot the data in `sample_data`. 

`sample_quick_test_job_array_grid.py` and `sample_quick_test_job_array_script.sh` are similar but run very quick jobs that are good for testing your environment setup. 

After you get the results (by default they show up in a folder called data), you can then use `python -m spinup.run plot <your data folder name>`. For example, if your folder name is "sac", then you should do `python -m spinup.run plot sac/`, make sure the name has that slash and it's not `python -m spinup.run plot sac`. 

## Changed:

1. plotting now supports specify colors with terminal argument. 
2. `sample_job_array_grid.py` now is updated, as long as settings are specified, they will be automatically added to experiment, so there will be less problem on forgetting to add setting. 

## Reference: 

Original SAC paper: https://arxiv.org/abs/1801.01290

OpenAI Spinup docs on SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

rlkit sac implementation: https://github.com/vitchyr/rlkit

Best practice for install conda and pip environments...
https://www.anaconda.com/blog/using-pip-in-a-conda-environment
"using pip only after all other requirements have been installed via conda is the safest practice."

## Acknowledgement 
Great thanks to Josh Achiam, the author of OpenAI Spinning Up for providing the spinup documentation and original codebase. Many thanks to hpc admin Zhiguo for his enormous support.


