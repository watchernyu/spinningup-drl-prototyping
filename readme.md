# Soft Actor-Critic Pytorch Implementation
Soft Actor-Critic Pytorch Implementation, based on the OpenAI Spinup documentation and some of its code base. This is a minimal, easy-to-learn and well-commented Pytorch implementation, and recommended to be studied along with the OpenAI Spinup Doc. This SAC implementation is based on the OpenAI spinningup repo, and uses spinup as a dependency. Target audience of this repo is Pytorch users (especially NYU students) who are learning DRL.

## Setup environment:
Make sure Go to a place on your machine where you can put python files. (for example, desktop or home, or create a folder), make sure you have Anaconda on your machine, then run the following commands in your terminal, which will create a conda environment called rl and then install for you:

If `source activate drl` does not work, you should try `conda activate drl`. You might want to run these commands one line at a time. Or if you konw what you are doing and you have setup certain required system packages correctly already, you can run multiple lines at a time. But please do spend a little time think about what you are doing by running that line, for example, `cd ..` will let you go to the parent folder's directory, so if you are in a location where you can download files, you run some commands, then you move to another location, and run another terminal, and you didn't change the current directory, and you also run the exact next command, without thinking about what is happening, then things will go wrong. 

## Note: download and installation can take some time.
It can take some time, especially for pytorch and tensorflow installation part. You can install them in the background, and when one step is finished, move on to the next. 

## First we create a new conda environment 
If `conda activate drl` does not work, then try `source activate drl`. 

If you are installing on the NYU Shanghai hpc, you first need to apply for an account, then you will use `ssh <netid>@hpc.shanghai.nyu.edu` to connect to the hpc, replace `<netid>` with your own netid. You can only access the hpc with this command when you are inside NYU network (using nyu wifi, or using nyu vpn). When you are connected, first run `module load anaconda3` so you have anaconda3. If you are using your own machine, you need to install anaconda on your own machine. 

On Linux and Windows (Note, for our test environments, Windows is not well supported, so will be problematic, consider use a ubuntu virtual machine):
```
conda create -n drl python=3.6
conda activate drl 
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.1 -c pytorch
```

On OSX: 
```
conda create -n drl python=3.6
conda activate drl 
conda install pytorch==1.3.1 torchvision==0.4.2 -c pytorch
```

## Then we download and install gym and mujoco-py. 
These are python packages, they provide the simulated environments where we test DRL agents. Make sure your terminal's current location is at a place where you can find, and where you have space to download some stuff. Note I have a `cd ..` in between installing these 2 packages, please don't install a package inside the folder of another. 
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
cd ..
git clone https://github.com/openai/mujoco-py
cd mujoco-py
pip install -e . --no-cache
cd ..
```

## Set up MuJoCo 
Openai gym has a number of environments to test on, we want to test on a list of robotic environments that are called MuJoCo environments. For these to work, we also need to download the MuJoCo physics engine. Go to this website `https://www.roboti.us/index.html`, and download the mujoco files for your operating system. For example, if you are on mac, click `mujoco200 macos`. We need to put these files to the correct location so that the python package `mujoco-py` can work, and then we can use those environments in `gym`. You will also need a license, ask your TA for the license. The instructions are given on this page `https://github.com/openai/mujoco-py`, basically, if you use linux or mac, (copy-pasted from that page), Unzip the downloaded `mujoco200` directory into `~/.mujoco/mujoco200`, and place your license key (the `mjkey.txt` or `authorized_keys`) in the folder `~/.mujoco/`. So in the end, under the folder `~/.mujoco/`, you should have a mujoco key file, and then also the folder `mujoco200`. 

## Test MuJoCo
Now before we move on, we want to test if MuJoCo works. Run python (make sure you are still in that drl virtual env), after you entered python:
```
import gym
import mujoco_py
```
You will see some warning messages, that is ok. But if you see an error, then something is wrong. If you do not see an error, only warnings, or no warnings, then proceed to initialize a gym MuJoCo environment: 
```
e = gym.make('Ant-v2')
e.reset()
```
The output should be a numpy array of numbers. If you can reach this step, then your gym and MuJoCo part should be correctly installed. For a better understanding of gym, look at this page `https://gym.openai.com/`. 

You might run into problems, you might try consult the openai gym github page  or the mujoco-py github page https://github.com/openai/mujoco-py. They have a list of known problems and potential solutions. 

## Download and install this particular repository. 
On mac, you first need to install openmpi with brew: `brew install openmpi`. If you are still in python from last step, then use `quit()`, or use the shortcut Ctrl+D to exit python and return to your terminal. We will now download and install some other packages. This repo is based on the openai spinup repo. We have added the pytorch version of the SAC algorithm and some hpc sample scripts etc. Make sure your terminal's current location is not inside a package folder, you can use `cd ..` to move to the parent folder. Now run the commands to download and install this repo:

```
git clone https://github.com/watchernyu/spinningup-drl-prototyping.git
cd spinningup-drl-prototyping
pip install numpy==1.16.4
pip install tensorflow==1.12.0
pip install seaborn==0.8.1
pip install -e .
cd ..
```

## Test SAC 
Now you have both the environment and the pytorch code for an SAC agent, make sure your current location is at the folder that contains `spinningup-drl-prototyping`, and run the following command for a quick test: 
```
cd spinningup-drl-prototyping/spinup/algos/sac_pytorch
python sac_adapt.py --hid 4 --steps_per_epoch 1000 --epochs 2
```

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

## Try plotting
Go to the `sample_data` folder, and you can test the plotting functionality. Start a terminal and run the following commands (skip the first line if you are already in the virtualenv):

```
conda activate drl
python -m spinup.run plot alg1data/alg1_ant-v2/ alg1data/alg1_halfcheetah-v2/ alg2data/alg2_ant-v2/ alg2data/alg2_halfcheetah-v2/
python -m spinup.run plot -s 10 alg1data/alg1_ant-v2/ alg2data/alg2_ant-v2/ --legend algorithm1 algorithm2 --color tab:orange tab:blue --value Performance --xlabel timestep --ylabel Performance
python -m spinup.run plot -s 10 alg1data/alg1_ant-v2/ alg2data/alg2_ant-v2/ --legend alg1 alg2 --color red blue --value AverageQ1Vals --xlabel timestep --ylabel QValue
```

Notice how you can change things such as color, label, legend with different optional arguments. Our plotting method is based on the Spinningup plot function, which is documented here: https://spinningup.openai.com/en/latest/user/plotting.html. You can also check the source code to see what options are available. 

## Setup on HPC
After you login to the HPC, you will now be on a login node, we will download and install python packages on this node, then test somewhere else. First load anaconda3 and some other modules:
```
module load anaconda3
module load cuda/9.0 glfw/3.3 gcc/7.3 mesa/19.0.5 llvm/7.0.1
```
Then you can proceed to perform the same installation process. Except that you need the linux mujoco files. You can install filezilla: https://filezilla-project.org/ and use it to transfer files between your machine and the hpc. To connect to the hpc via Filezilla, open the site manager (its icon is typically at the top left corner of the window), add a new site: set Host = `hpc.shanghai.nyu.edu`, port = `22`, protocol: `SFTP`, Logon Type:`Normal`, and enter your NYU credentials. Later on you can also add bookmarks so that you can easily go to certain commonly-used locations on the hpc. You can also use quickconnect. 

After you installed everything, use this command to start an interactive shell on a non-login node: 
`srun -p aquila --pty --mem  5000 -t 0-01:00 bash`, now you will be in one of the non-login nodes, these nodes don't have internet connection so you cannot download stuff, but you can perform test here, now use `source deactive` to deactivate your virtual environment, then you can active your environment, and then perform the tests (import mujoco, run sac etc.) here. 

Note: if you are submitting jobs using sbatch, make sure you deactivate your environment, or simply log out and log in again before submitting the job. 


## Changed:

Now by default use `sac_adapt_fast.py`, which takes one udpate after each data collection. This is more consistent with SAC paper and might be more robust. 

## Reference: 

Original SAC paper: https://arxiv.org/abs/1801.01290

OpenAI Spinup docs on SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

rlkit sac implementation: https://github.com/vitchyr/rlkit

Best practice for install conda and pip environments...
https://www.anaconda.com/blog/using-pip-in-a-conda-environment
"using pip only after all other requirements have been installed via conda is the safest practice."

## Fix problems
It is highly likely you will run into some problems when installing on your own machine, most of the time, this is because some python, or (more commonly) non-python packages are missing. Solutions to some of these problems can be found on the github pages or the documentation site of the packages we use. But here we also list a number of common issues and solutions. 

### mac gcc error
Some **mac** users will run into gcc error. Typically it will first tell you to install gcc with brew. From past experience, it seems `gcc-6` works, so first try install with brew with the command `brew install gcc@6`, if you don't have brew, you will need to first install brew. If you run into a brew installation error, then try uninstall your brew and then reinstall brew, and then try `brew install gcc@6` again. After this step, you should have `gcc-6` now, but things might still not work, because your `gcc` command is mapped to the wrong gcc version. Check your gcc version wtih the command: `gcc --version`, gcc-6 is the version that works. gcc-4 and gcc-7 seem to fail (not sure why gcc 7 works on linux but fails on mac??). Likely you will see sth that is not gcc-6, so now you want to change your default gcc. This can be done with the command:

```
cd /usr/local/bin
rm gcc
ln -s gcc-6 gcc
```

Essentially you create a symbolic link so that your `gcc` points to `gcc-6`. If you are interested, here is a tutorial on how symbolic link works: https://www.youtube.com/watch?v=-edTHKZdkjo start from 4:25. 

### mac openmpi missing
If you are seeing error related to openmpi, go to this page https://spinningup.openai.com/en/latest/user/installation.html and then follow the instructions on `brew install openmpi`. 

### windows missing vcvarsall.bat
This is likely due to lack of C++ compile tools, look for the gtatiya response: 
https://github.com/openai/mujoco-py/issues/253

Essentially you want to install the following file: 
Install Microsoft Visual C++ Build Tools 2015: https://download.microsoft.com/download/5/f/7/5f7acaeb-8363-451f-9425-68a90f98b238/visualcppbuildtools_full.exe?fixForIE=.exe

### hpc missing patchelf
Simply install patchelf.

### hpc: an error telling you to add a line to .bashrc
Simply do what the instructions tell you. Find that file and add that line. You can use export or nano, if you don't know what these are, you can use filezilla to connect, and once you connect, you can try to find that file (the file is located at `/gpfsnyu/home/netid`, which is precisely the default location after you connected to the hpc). This is your home directory on the hpc, and in the terminal you can use `cd ~` to go to the same place. You can send this file back to your local machine, edit it with any text editor, and then send it back to the hpc to overwrite it. 

## Extra stuff: optimize your workflow
### ssh key pair for easy login
Make a ssh key pair so that you don't type your password everytime you login to hpc: https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys-on-ubuntu-1604 or https://www.youtube.com/watch?v=vpk_1gldOAE

## Acknowledgement 
Great thanks to Josh Achiam, the author of OpenAI Spinning Up for providing the spinup documentation and original codebase. Many thanks to hpc admin Zhiguo for his enormous support.
