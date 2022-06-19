#!/usr/bin/env python3

"""
Plots a time-series of snapshots from each simulation.
"""

from CPM.cpm import CPM
import matplotlib.pyplot as plt
import _pickle as cPickle
import bz2
import os
import sys
import numpy as np


if __name__ == "__main__":

    if not os.path.exists("../results"):
        os.mkdir("../results")

    if not os.path.exists("../results/time_series_plots"):
        os.mkdir("../results/time_series_plots")

    if not os.path.exists("../results/time_series_plots/soft"):
        os.mkdir("../results/time_series_plots/soft")


    if not os.path.exists("../results/time_series_plots/stiff"):
        os.mkdir("../results/time_series_plots/stiff")

    if not os.path.exists("../results/time_series_plots/scrambled"):
        os.mkdir("../results/time_series_plots/scrambled")



    iter_i = int(sys.argv[1])

    typ = "scrambled" #specify here the sub-directory of simulation outputs that one wants to plot for.

    I_save_sparse = cPickle.load(bz2.BZ2File("../results/%s/%d.pbz2"%(typ,iter_i), 'rb'))

    I_save = np.array([Iss.todense() for Iss in I_save_sparse])


    c_types = np.zeros(23,dtype=int)
    c_types[1:] = 1
    c_types[9:] = 2
    c_types[17:] = 3
    cpm = CPM({"T":15})
    cpm.n_cells = I_save.max()
    cpm.c_types = c_types[1:]
    t = np.linspace(0,len(I_save)-1,5).astype(int)

    fig, axs = plt.subplots(1,len(t))
    summed_I_save = np.sum(I_save, axis=0)
    x_ids,y_ids = np.nonzero(summed_I_save)
    xlim = (x_ids.min(),x_ids.max())
    ylim = (y_ids.min(),y_ids.max())

    axs = axs.ravel()
    for i, ti in enumerate(t):
        ax = axs[i]
        I = I_save[ti][xlim[0]:xlim[1]+1,ylim[0]:ylim[1]+1]
        ax.imshow(cpm.generate_image(I, res=8, col_dict={1: "red", 2: "blue", 3: "green"},background=np.array([0,0,0,0.])))
        ax.axis("off")
    fig.show()
    fig.savefig("../results/time_series_plots/%s/%d.pdf"%(typ,iter_i),dpi=300)