"""
Use this to save plot as a file instead of show plot: --save-name <filename>
Use this to change legend location (for example, to lower right): --legend-loc=\'lower right\'
Use this to have no legend: --no-legend
Use this to specify legend names (for example, 4 curves): --legend SAC SAC+ERE SAC+PER SAC+ERE+PER

###### simply run this to generate the URRL4 performance plots
"""
import os
from spinup.utils.plot import make_plots

# what data column is used as your xaxis
xaxis = 'TotalEnvInteracts'
# what data column for your y axis, 'Performance' is default value, it means either 'AverageTestEpRet' or 'AverageEpRet'
value = 'Performance'
smooth = 10
# e.g. 'best' 'lower right'
legend_loc = 'lower right'
label_font_size = 16
save_path = '../plots/alg_test_compare' # where your plot is saved to
data_path_prefix = '../sample_data' # where the data is

legend = ['Alg1', 'Alg2']
color = ['tab:orange', 'tab:green']

if not os.path.exists(save_path):
    os.mkdir(save_path)

# envnames = ['halfcheetah','walker2d','ant','hopper','humanoid']
envnames = ['halfcheetah','ant']
alg_folder_names = ['alg1data', 'alg2data']
alg_names = ['alg1','alg2'] # you might want to name your folder the same name as your algorithms

for e in envnames: # make one plot for each environment
    datapaths = []
    for alg_i in range(2):
        folder_name = alg_folder_names[alg_i]
        alg_name = alg_names[alg_i]
        dataname = '%s_%s-v2/' % (alg_name, e)
        datapath = os.path.join(data_path_prefix, folder_name, dataname)
        datapaths.append(datapath)
    save_name = 'sample_plot_savename_%s' % (e)
    make_plots(datapaths, xaxis=xaxis, values=value, legend_loc=legend_loc,
               label_font_size=label_font_size, save_path=save_path, save_name=save_name)
