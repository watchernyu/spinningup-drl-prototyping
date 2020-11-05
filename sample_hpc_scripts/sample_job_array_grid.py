from spinup.utils.run_utils import ExperimentGrid
from spinup.algos.sac_pytorch.sac_adapt_fast import sac_adapt_fast as function_to_run ## here make sure you import correct function
import time

"""
always first change the experiment name first 
change it to a name that makes sense to your experiment, 
and make sure you import the correct function to run
you can do a quick test on your own machine before you upload to hpc
|||||||||||||||||||||||||||||||||
VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
"""

EXPERIMENT_NAME = 'SAC'

"""
always change the experiment name first 
"""

# this is where your progress files will show up, it is a relative path
# relative to this python file
save_data_dir = '../data'

if __name__ == '__main__':
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()
    """
    if you are adding more settings, try to add them in a consistent manner in terms of order
    for example, first learning rate, then batch size
    by default, wherever you put environment, the environment name will arrive at the end of the file name. seed will not
    show up by default. Your other settings show up in the file name in the same order as they are in the setting_names list
    also make sure in the .sh file you changed the number of jobs in the array
    if you can't count, then simply run this program and quickly stop it on your machine
    it will print out how many settings there are
    then change that line to: 
    #SBATCH --array=0-<number of jobs - 1>
    """
    setting_names = ['env_name',
                     'seed', 'lr']
    settings = [['Humanoid-v2', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Swimmer-v2', 'Walker2d-v2'],
               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0.3, 0.5]]
    whether_add_to_savename = [True, False, True]
    setting_savename_prefix = ['', '', 'learning_rate']

    n_setting = len(setting_names)
    assert_correct = (len(settings) == n_setting and len(whether_add_to_savename)==n_setting and len(setting_savename_prefix)==n_setting)
    if not assert_correct:
        print("\nASSERTION FAILED, NUMBER OF SETTINGS DON'T MATCH!!!!\n")
        quit()

    ##########################################DON'T NEED TO MODIFY#######################################
    ##############SIMPLY DON'T MODIFY ANYTHING AFTER THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING########
    ## this block will assign a certain set of setting to a "--setting" number
    ## basically, maps a set of settings to a hpc job array id
    total = 1
    for sett in settings:
        total *= len(sett)

    print("##### TOTAL NUMBER OF JOBS: %d #####" % total)
    if args.setting >= (total):
        print("setting number is larger than total number of setting, quit now")
        quit()

    def get_setting(setting_number, total, settings, setting_names):
        indexes = []  ## this says which hyperparameter we use
        remainder = setting_number
        for setting in settings:
            division = int(total / len(setting))
            index = int(remainder / division)
            remainder = remainder % division
            indexes.append(index)
            total = division
        actual_setting = {}
        for j in range(len(indexes)):
            actual_setting[setting_names[j]] = settings[j][indexes[j]]
        return indexes, actual_setting

    indexes, actual_setting = get_setting(args.setting, total, settings, setting_names)

    eg = ExperimentGrid(name=EXPERIMENT_NAME)
    # use eg.add to add parameters in the settings or add parameters tha apply to all jobs
    # we now automated this part, as long as you added settings correctly into the arrays at the start of this program
    # they should be added to experiment automatically
    for i in range(len(actual_setting)):
        setting_name = setting_names[i]
        if setting_name != 'env_name' and setting_name != 'seed':
            eg.add(setting_name, actual_setting[setting_name], setting_savename_prefix[i], whether_add_to_savename[i])

    eg.add('env_name', actual_setting['env_name'], '', True)
    eg.add('seed', actual_setting['seed'])

    eg.run(function_to_run, num_cpu=args.cpu, data_dir=save_data_dir)

    print('\n###################################### GRID EXP END ######################################')
    print('total time for grid experiment:',time.time()-start_time)
