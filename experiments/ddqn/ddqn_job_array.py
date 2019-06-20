from spinup.utils.run_utils import ExperimentGrid
from spinup.algos.ddqn_pytorch.double_dqn import double_dqn
import time

if __name__ == '__main__':
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()
    ## MAKE SURE ALPHA IS ADDED, MAKE SURE EACH SETTING IS ADDED
    ## MAKE SURE exp name is change, make sure used correct sac function

    setting_names = ['env_id', 'seed']
    settings = [['BreakoutNoFrameskip-v4', 'DemonAttackNoFrameskip-v4', 'AssaultNoFrameskip-v4', 'AtlantisNoFrameskip-v4'],
               [0, 1, 2, 3, 4]]

##########################################DON'T NEED TO MODIFY#######################################
    ## this block will assign a certain set of setting to a "--setting" number
    ## basically, maps a set of settings to a hpc job array id
    total = 1
    for sett in settings:
        total *= len(sett)

    print("total: ", total)

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
####################################################################################################

    eg = ExperimentGrid(name='DDQN')

    eg.add('env_id', actual_setting['env_id'], '', True)
    eg.add('seed', actual_setting['seed'])

    eg.run(double_dqn, num_cpu=args.cpu)

    print('\n###################################### GRID EXP END ######################################')
    print('total time for grid experiment:',time.time()-start_time)
