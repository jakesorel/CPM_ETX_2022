#!/usr/bin/env python3

"""
For analysis of sorting efficacy under different lambda_P values for XEN cells.

A script to run the CPM model, given a set of bootstrap-sampled adhesion matrices.

Runs for a set of lambda_P multipliers, in series. With the same bootstrap-sampled adhesion matrix.

Run from the command-line:

e.g. python run_softstiff.py 72

where 72 defines the bootstrap adhesion matrix that is to be used for parameterising the CPM.

See run_scripts/make_adhesion_matrices.py for details on the bootstraping procedure.
"""
import os
import sys
import time

import numpy as np

from CPM.cpm import CPM

if __name__ == "__main__":

    # Establish the directory structure for saving.
    if not os.path.exists("../results"):
        os.mkdir("results")

    if not os.path.exists("../results/variable_soft"):
        os.mkdir("../results/variable_soft")

    # Get the index of the boostrap adhesion matrix to be used to parameterise the CPM. From the command-line.
    iter_i = int(sys.argv[1])

    # Define the range of lambda_P multiples to be spanned across in the parameter scan.
    lambda_P_mult_range = np.flip(np.linspace(0.2, 1, 9))
    for i, lpm in enumerate(lambda_P_mult_range):
        dir_name = "../results/variable_soft/%.2f" % lpm
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    # Define the parameters.
    A0 = 30
    P0 = 0
    lambda_A = 1
    lambda_P = 0.2
    b_e = -0.5

    # Define the W-matrix. Needed, given the architecture, but actually not used in practice, as is replaced by the
    # bootstrapped adhesion matrices.
    W = np.array([[b_e, b_e, b_e, b_e],
                  [b_e, 1.911305, 0.494644, 0.505116],
                  [b_e, 0.494644, 2.161360, 0.420959],
                  [b_e, 0.505116, 0.420959, 0.529589]]) * 6.02

    # Run the first simulation for the lowest lambda_P multiplier.
    lpm = lambda_P_mult_range[0]

    params = {"A0": [A0, A0, A0],
              "P0": [P0, P0, P0],
              "lambda_A": [lambda_A, lambda_A, lambda_A],
              "lambda_P": [lambda_P, lambda_P, lambda_P * lpm],
              "W": W,
              "T": 15}
    cpm = CPM(params)
    cpm.make_grid(100, 100)
    cpm.generate_cells(N_cell_dict={"E": 8, "T": 8, "X": 6})
    cpm.make_init("circle", np.sqrt(params["A0"][0] / np.pi) * 0.8, np.sqrt(params["A0"][0] / np.pi) * 0.2)

    #import the bootstrapped adhesion matrix.
    adhesion_vals_full = np.load("../bootstrap_samples/adhesion_matrices/%i.npz" % iter_i).get("adhesion_vals")
    adhesion_vals_full[0] = b_e * cpm.lambda_P
    adhesion_vals_full[:, 0] = b_e * cpm.lambda_P
    adhesion_vals_full[0, 0] = 0
    cpm.J = -adhesion_vals_full * 6

    cpm.get_J_diff()
    t0 = time.time()
    cpm.simulate(int(1e7), int(1000), initialize=True, J0=-8)

    cpm.save_simulation("../results/variable_soft/%.2f" % lpm, str(iter_i))

    #Run with the rest of the lambda_P multipliers. Uses the same initialisation, such that the simulations are fully
    # comparable
    for i in range(len(lambda_P_mult_range)):
        lpm = lambda_P_mult_range[i + 1]

        params2 = {"A0": [A0, A0, A0],
                   "P0": [P0, P0, P0],
                   "lambda_A": [lambda_A, lambda_A, lambda_A],
                   "lambda_P": [lambda_P, lambda_P, lambda_P * lpm],
                   "W": W,
                   "T": 15}
        cpm2 = CPM(params2)
        cpm2.make_grid(100, 100)
        cpm2.generate_cells(N_cell_dict={"E": 8, "T": 8, "X": 6})

        #Use the same initialisation as the first simulation.
        cpm2.I0 = cpm.I_save[0]
        cpm2.I = cpm2.I0.copy()
        cpm2.n_cells = cpm.n_cells
        cpm2.assign_AP()

        #copy over the bootstrapped-adhesion matrix.
        cpm2.J = cpm.J.copy()
        cpm2.J[0] = -6 * b_e * cpm2.lambda_P
        cpm2.J[:, 0] = -6 * b_e * cpm2.lambda_P
        cpm2.J[0, 0] = 0
        cpm2.J_diff = cpm.J_diff.copy()
        cpm2.simulate(int(1e7), int(1000), initialize=False)
        cpm2.save_simulation("../results/variable_soft/%.2f" % lpm, str(iter_i))
