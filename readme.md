# Soft Actor-Critic Pytorch Implementation
Soft Actor-Critic Pytorch Implementation, based on the OpenAI Spinup documentation and some of its code base. This is a minimal, easy-to-learn and well-commented Pytorch implementation, and recommended to be studied along with the OpenAI Spinup Doc. This SAC implementation is based on the OpenAI spinningup repo, and uses spinup as a dependency. Target audience of this repo is Pytorch users (especially NYU students) who are learning Soft Actor-Critic algorithm. 

## Setup environment:
To use the code you should first download this repo, and then install this repo with the same method to install the original spinup repo. (So you are using the same method, but you should install this repo, not the original repo.)

the spinup documentation is here, you should read it to make sure you know the procedure: https://spinningup.openai.com/en/latest/user/installation.html

The only difference in installation is you want to install this repo, instead of the original repo, don't download the original repo, use this repo please. When you are ready to install this in a virtualenv (and don't forget to actually enter your virtualenv) you should first clone this repo onto your machine, enter the repo folder, and then use the pip install command: 

```
pip install -e .
```

The Pytorch version used is: 1.2, install pytorch:
https://pytorch.org/

tensorflow might be needed, install version 1.12.0

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

## Changed:

1. plotting now supports specify colors with terminal argument. 
2. `sample_job_array_grid.py` now is updated, as long as settings are specified, they will be automatically added to experiment, so there will be less problem on forgetting to add setting. 

## Reference: 

Original SAC paper: https://arxiv.org/abs/1801.01290

OpenAI Spinup docs on SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

rlkit sac implementation: https://github.com/vitchyr/rlkit

## Acknowledgement 
Great thanks to Josh Achiam, the author of OpenAI Spinning Up for providing the spinup documentation and original codebase. Many thanks to hpc admin Zhiguo for his enormous support.
