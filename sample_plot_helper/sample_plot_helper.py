"""
Plot helper: use this kind of file to generate plotting commands, to save time
Use this to save plot as a file instead of show plot: --save-name <filename>
Use this to change legend location (for example, to lower right): --legend-loc=\'lower right\'
Use this to have no legend: --no-legend
Use this to specify legend names (for example, 4 curves): --legend alg1 alg2 alg3 alg4
"""

"""
This sample plot helper file can be used to plot the data in the sample data folder
assume data is organized such that they look like what's in the sample data folder
simple change the code in this file to suit your need for plotting
"""

basic_str = 'python -m spinup.run plot -s 10 --save-name '

envs = ['halfcheetah-v2', 'ant-v2' ]

for i in range(len(envs)):
    e = envs[i]
    savename = 'sample_plot_savename_' + e[:-3]+ ' '
    s_alg1 = 'alg1/alg1_' + e + '/ '
    s_alg2 = 'alg2/alg2_' + e + '/ '

    plot_str = basic_str + savename + s_alg1 +s_alg2 + '--legend Alg1 Alg2 ' +' --legend-loc \'lower right\' --color tab:orange tab:green '
    print(plot_str)

"""
run this program to obtain the plot command
then simply go to the folder that contains all the data, and then copy paste into your terminal
"""
