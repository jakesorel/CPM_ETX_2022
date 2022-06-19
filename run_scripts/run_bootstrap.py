#!/usr/bin/env python3

"""
A script to run the CPM model, given a set of bootstrap-sampled adhesion matrices.

Run from the command-line:

e.g. python run_softstiff.py 72

where 72 defines the bootstrap adhesion matrix that is to be used for parameterising the CPM.

See run_scripts/make_adhesion_matrices.py for details on the bootstraping procedure.
"""
import numpy as np
from CPM.cpm import CPM
import matplotlib.pyplot as plt
import time
import os
import sys


if __name__ == "__main__":

    #Set up directory architecture.
    if not os.path.exists("../results"):
        os.mkdir("../results")

    if not os.path.exists("../results/plots"):
        os.mkdir("../results/plots")

    if not os.path.exists("../results/stiff"):
        os.mkdir("../results/stiff")

    if not os.path.exists("../results/scrambled"):
        os.mkdir("../results/scrambled")

    if not os.path.exists("results/plots/stiff"):
        os.mkdir("results/plots/stiff")

    if not os.path.exists("../results/plots/scrambled"):
        os.mkdir("../results/plots/scrambled")


    #Index of the bootstrapped adhesion-matrix. From the command-line.
    iter_i = int(sys.argv[1])

    #Set up the parameters
    A0 = 30
    P0 = 0
    lambda_A = 1
    lambda_P = 0.2
    b_e = -0.5

    # Define the W-matrix. Needed, given the architecture, but actually not used in practice, as is replaced by the
    # bootstrapped adhesion matrices.
    W = np.array([[b_e,b_e,b_e,b_e],
                  [b_e,1.911305,0.494644,0.505116],
                  [b_e,0.494644,2.161360,0.420959],
                  [b_e,0.505116,0.420959,0.529589]])*6.02


    params = {"A0":[A0,A0,A0],
              "P0":[P0,P0,P0],
              "lambda_A":[lambda_A,lambda_A,lambda_A],
              "lambda_P":[lambda_P,lambda_P,lambda_P],
              "W":W,
              "T":15}
    cpm = CPM(params)
    cpm.make_grid(100,100)
    cpm.generate_cells(N_cell_dict={"E": 8, "T": 8,"X":6})
    cpm.make_init("circle", np.sqrt(params["A0"][0] / np.pi) * 0.8, np.sqrt(params["A0"][0] / np.pi) * 0.2)

    #Import the bootstrapped adhesion values, derived from the AFM data.
    adhesion_vals_full = np.load("../adhesion_matrices/%i.npz" % iter_i).get("adhesion_vals")
    adhesion_vals_full[0] = b_e*cpm.lambda_P
    adhesion_vals_full[:,0] = b_e*cpm.lambda_P
    adhesion_vals_full[0,0] = 0
    cpm.J = -adhesion_vals_full * 6
    cpm.get_J_diff()
    cpm.simulate(int(1e7), int(1000), initialize=True, J0=-8)
    cpm.save_simulation("results/stiff",str(iter_i))

    #Save the last frame of the CPM simulation as an image.
    fig, ax = plt.subplots()
    ax.imshow(cpm.generate_image(cpm.I, res=8, col_dict={1: "red", 2: "blue", 3: "green"}))
    ax.axis("off")
    fig.savefig("results/plots/stiff/%d.pdf"%iter_i,dpi=300)

    #Repeat the above, but with scrambled parameters. With the same initialisation as the above simulation, for fair
    # comparison
    cpm3 = CPM(params)
    cpm3.make_grid(100, 100)
    cpm3.generate_cells(N_cell_dict={"E": 8, "T": 8, "X": 6})
    #Copy over the initialisation from the first simulation.
    cpm3.I0 = cpm.I_save[0]
    cpm3.I = cpm3.I0.copy()
    cpm3.n_cells = cpm.n_cells
    cpm3.assign_AP()

    #Import the bootstrapped adhesion values scrambled, derived from the AFM data.
    adhesion_vals_full = np.load("../adhesion_matrices_scrambled/%i.npz" % iter_i).get("adhesion_vals")
    adhesion_vals_full[0] = b_e*cpm3.lambda_P
    adhesion_vals_full[:,0] = b_e*cpm3.lambda_P
    adhesion_vals_full[0,0] = 0
    cpm3.J = -adhesion_vals_full * 6
    cpm3.get_J_diff()
    cpm3.simulate(int(1e7), int(1000), initialize=False)
    cpm3.save_simulation("results/scrambled", str(iter_i))

    #Save the last frame of the CPM simulation as an image.
    fig, ax = plt.subplots()
    ax.imshow(cpm3.generate_image(cpm3.I, res=8, col_dict={1: "red", 2: "blue", 3: "green"}))
    ax.axis("off")
    fig.savefig("results/plots/scrambled/%d.pdf" % iter_i, dpi=300)

