#!/usr/bin/env python3

"""
A script to generate the bootstrap matrices of adhesion strengths, by resampling the AFM measurements of cell-by-cell
adhesion strength (found in raw_data/adhesion_dict.json).
"""

import json
import os

import numpy as np

if __name__ == "__main__":

    # number of bootstrap iterations to be used in the ensemble.
    n_iter = 500

    # where the adhesion data is stored. This has been reformatted as a json dictionary.
    adhesion_dict = json.load(open("../raw_data/adhesion_dict.json"))

    # number of cells in the CPM simulation. nE is the number of ES, nT the number of TS, and nX the number of XEN.
    nE, nT, nX = 8, 8, 6

    # Coding between cell-type names and cell_type indices. 0 -> ES, 1 -> TS, 2 -> XEN.
    pair_names = {(0, 0): "ES-ES",
                  (1, 1): "TS-TS",
                  (0, 1): "TS-ES",
                  (1, 0): "TS-ES",
                  (0, 2): "XEN-ES",
                  (2, 0): "XEN-ES",
                  (1, 2): "XEN-TS",
                  (2, 1): "XEN-TS",
                  (2, 2): "XEN-XEN"}


    def sample(pair):
        """
        Given a pair of cell type indices e.g. (0,1), randomly sample the adhesion dictionary once. Ignore nan values.
        :param pair:
        :return:
        """
        val = np.nan
        while np.isnan(val):
            val = np.random.choice(adhesion_dict[pair_names[pair]])
        return val


    def get_adhesion_matrix(nE, nT, nX):
        """
        Generate a (nE + nT + nX x nE + nT + nX) matrix of adhesion values by resampling the AFM adhesion data.
        :param nE: number of ES cells in the CPM simulation
        :param nT: number of TS cells in the CPM simulation
        :param nX: number of XEN cells in the CPM simulation
        :return: a (nE + nT + nX x nE + nT + nX) matrix of adhesion values by resampling the AFM adhesion data.
        """
        # number of cells is the sum of the number of cells of each type.
        nc = nE + nT + nX

        # establish a vector cell_types, where the first nE are 0, the next nT are 1, and the final nX are 2.
        c_types = np.zeros((nc), dtype=int)
        c_types[nE:nE + nT] = 1
        c_types[nE + nT:] = 2

        # Across the nc x nc matrix, randomly sample adhesion values from the AFM data.
        c_type1, c_type2 = np.meshgrid(c_types, c_types, indexing="ij")
        c_type_pairs = list(zip(c_type1.ravel(), c_type2.ravel()))
        adhesion_vals = list(map(sample, c_type_pairs))
        adhesion_vals = np.array(adhesion_vals).reshape((len(c_types), len(c_types)))
        adhesion_vals = np.triu(adhesion_vals, 1) + np.triu(adhesion_vals, 1).T
        return adhesion_vals


    def get_adhesion_matrix_scrambled(nE, nT, nX):
        """
        Same as *get_adhesion_matrix* apart from scrambles the order of cell-types, making cell-cell adhesion values and
        the cell-types indpendent.
        :param nE: number of ES cells in the CPM simulation
        :param nT: number of TS cells in the CPM simulation
        :param nX: number of XEN cells in the CPM simulation
        :return: a (nE + nT + nX x nE + nT + nX) SCRAMBLED matrix of adhesion values by resampling the AFM adhesion data
        """
        nc = nE + nT + nX

        c_types = np.zeros((nc), dtype=int)
        c_types[nE:nE + nT] = 1
        c_types[nE + nT:] = 2
        np.random.shuffle(c_types)

        c_type1, c_type2 = np.meshgrid(c_types, c_types, indexing="ij")
        c_type_pairs = list(zip(c_type1.ravel(), c_type2.ravel()))
        adhesion_vals = list(map(sample, c_type_pairs))
        adhesion_vals = np.array(adhesion_vals).reshape((len(c_types), len(c_types)))
        adhesion_vals = np.triu(adhesion_vals, 1) + np.triu(adhesion_vals, 1).T
        return adhesion_vals


    # Establish directory structure, for saving.
    if not os.path.exists("../bootstrap_samples"):
        os.mkdir("../bootstrap_samples")

    if not os.path.exists("../bootstrap_samples/adhesion_matrices"):
        os.mkdir("../bootstrap_samples/adhesion_matrices")

    if not os.path.exists("../bootstrap_samples/adhesion_matrices_scrambled"):
        os.mkdir("../bootstrap_samples/adhesion_matrices_scrambled")

    # Save adhesion matrices, and the corresponding scrambled ones, to file. In npz compressed format.
    nc = nE + nT + nX
    for i in range(n_iter):
        adhesion_vals = get_adhesion_matrix(nE, nT, nX)
        adhesion_vals_full = np.zeros((nc + 1, nc + 1))
        adhesion_vals_full[1:, 1:] = adhesion_vals
        np.savez("adhesion_matrices/%i.npz" % i, adhesion_vals=adhesion_vals_full)

    nc = nE + nT + nX
    for i in range(n_iter):
        adhesion_vals = get_adhesion_matrix_scrambled(nE, nT, nX)
        adhesion_vals_full = np.zeros((nc + 1, nc + 1))
        adhesion_vals_full[1:, 1:] = adhesion_vals
        np.savez("adhesion_matrices_scrambled/%i.npz" % i, adhesion_vals=adhesion_vals_full)
