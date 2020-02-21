import pandas as pd
import json
import os
import os.path as osp
import numpy as np

"""
python -m spinup.run hyper_search <files> -ae <start from which epoch>
make a file that can order the experiments in terms of their performance
use this to easily find good hyperparameters when doing hyperparameter search
upload this file when it's ready don't use it again lol 
"""

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def compute_hyper(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, no_legend=False,
                  legend_loc='best', color=None, linestyle=None, font_scale=1.5,
                  label_font_size=24, xlabel=None, ylabel=None, after_epoch=0, no_order=False,
                  **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    # print("columns", data.columns)
    unique_names = data[condition].unique() ## these are the experiment names

    n_settings = len(unique_names)
    score_list = np.zeros(n_settings)
    std_list = np.zeros(n_settings)
    print(score_list)
    for i in range(n_settings):
        un = unique_names[i]
        print("\nunique name: ",un)
        exp_data = data.loc[data[condition] == un] ## the data related to this experiment
        # average_test_epret = exp_data['AverageTestEpRet'].values
        # print(average_test_epret.shape)

        # final performance data only concern the last few epoches
        final_performance_data = exp_data.loc[exp_data['Epoch'] >= after_epoch]
        average_test_epret_final = final_performance_data['AverageTestEpRet'].values
        mean_score = average_test_epret_final.mean()
        std_score = average_test_epret_final.std()
        score_list[i] = mean_score
        std_list[i] = std_score
        epoch_reached = final_performance_data['Epoch'].max()
        if np.isnan(mean_score):
            print('n/a')
        else:
            print('total epoch: %d, score: %.2f' % (epoch_reached,mean_score))
    """
    here we want to give an ordering of the hyper-settings, so that we can know
    which ones are good hyper-parameters 
    """
    sorted_index =np.flip(np.argsort(score_list))
    if no_order:
        sorted_index = np.arange(len(sorted_index))
    for i in range(n_settings):
        setting_index = sorted_index[i]
        print('%s\t%.1f\t%.1f' % (unique_names[setting_index], score_list[setting_index], std_list[setting_index]))

def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
                performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
                exp_data.insert(len(exp_data.columns), 'Unit', unit)
                exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
                exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
                exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
                datasets.append(exp_data)
            except Exception as e:
                print(e)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == '/':
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split('/')[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not (legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data

def compare_performance(all_logdirs, legend=None, xaxis=None, values=None, count=False,
                        font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', no_legend=False,
                        legend_loc='best', after_epoch=0,
                        save_name=None, xlimit=-1, color=None, linestyle=None, label_font_size=24,
                        xlabel=None, ylabel=None,
                        no_order=False):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
    for value in values:
        compute_hyper(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, no_legend=no_legend,
                      legend_loc=legend_loc,
                      estimator=estimator, color=color, linestyle=linestyle, font_scale=font_scale,
                      label_font_size=label_font_size,
                      xlabel=xlabel, ylabel=ylabel, after_epoch=after_epoch, no_order=no_order)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--after-epoch', '-ae', type=int, default=0)
    parser.add_argument('-no', '--no-order', action='store_true')

    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

        after-epoch: if > 0 then when computing an algorithm's "score", 
            we will use the average of test returns after a certain epoch number
            
        no-order: have this option so it doesn't print setting names in order of performance
    """

    compare_performance(args.logdir, args.legend, args.xaxis, args.value, args.count,
                        smooth=args.smooth, select=args.select, exclude=args.exclude,
                        estimator=args.est, after_epoch=args.after_epoch, no_order=args.no_order)


if __name__ == "__main__":
    main()

