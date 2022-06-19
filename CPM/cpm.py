#!/usr/bin/env python3

"""
This module defines the CPM class, the major workhorse of the code-base.
"""
import _pickle as cPickle
import bz2
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors
from scipy import sparse

from sample import Sample


class CPM:
    """
    The CPM class.

    Defines the geometries of cells and their corresponding contacts, then performs Metropolis-Hastings optimisation of
    an energy functional to establish the low-energy conformations of the in-silico tissues.

    Details are provided in the Methods and Supplementary Information of Bao et al., Nature Cell Biology 2022.
    """
    def __init__(self, params=None):
        """
        Initialisation of the CPM class.
        @param params: A dictionary of parameters.
        """
        assert params is not None, "Specify params"
        self.params = params
        self.num_x, self.num_y = None, None
        self.I = None
        self.boundary_mask = None

        self.A, self.P, self.A0, self.P0 = None, None, None, None
        self.lambda_P, self.lambda_A = None, None

        self.Moore, self.perim_neighbour = None, None
        self.define_neighbourhood()
        self.sample = Sample(self)
        self.do_steps = self.sample.do_steps

        self.neighbour_options = np.array([[1, 0],
                                           [-1, 0],
                                           [0, 1],
                                           [0, -1]])

    def define_neighbourhood(self):
        """
        Define the multiple neighbourhoods used in calculations.

        Moore neighbourhood (self.Moore) is the 3x3 set of x and y shifts centred on (0,0).

        Perim_neighbour (self.perim_neighbour) is the Moore neighbourhood, minus the central position.
        """
        i_, j_ = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
        self.Moore = np.array([i_.ravel(), j_.ravel()]).T
        i_2, j_2 = np.delete(i_.ravel(), 4), np.delete(j_.ravel(), 4)
        self.perim_neighbour = np.array([i_2, j_2]).T

    def make_grid(self, num_x=300, num_y=300):
        """
        Makes the initial square grid, where the CPM formulation of cellular objects and tissues is housed.

        Initialises I with zeros (ints).
        I is the (num_x x num_y) matrix of ints. Each int is the index corresponds to a cell. Thus each index is Moore
        contiguous within I.

        @param num_x: Number of pixels in the x-dimension of I.
        @param num_y: Number of pixels in the y-dimension of I.
        """
        self.num_x, self.num_y = num_x, num_y
        self.I = np.zeros([num_x, num_y]).astype(int)

    def generate_cells(self, N_cell_dict):
        """
        Fill the CPM simulation with a set of cells, defining a tissue that will be optimised under the M-H algorithm,
        via minimising an energy functional.
        Defines self.c_types, a list of cell_types, which are indexed 0,1,2,... via the N_cell_dict. 0 is saved for
        'medium' psuedo-cell, whereas 1,2,... are the remaining cell-types.

        E.g. if N_cell_dict = {"E": 8, "T": 8,"X":6}; c_type = 1 <--> "E"

        Additionally, defines the number of cells, self.n_cells.
        And the cell_ids, which count up from 1. 0 is reserved for the 'medium' psuedo-cell.

        @param N_cell_dict: A dictionary of cell-types and corresponding numbers of each cell.
         e.g. {"E": 8, "T": 8,"X":6}
        @return:
        """
        self.c_types = []
        for type_i, (Type, n_i) in enumerate(N_cell_dict.items()):
            for i in range(n_i):
                self.c_types += [type_i + 1]
        self.n_cells = len(self.c_types)
        self.cell_ids = np.arange(1, self.n_cells + 1)
        self.set_cell_params()

    def set_cell_params(self):
        """
        Converts the parameters of the energy functional, prescribed in the dictionary self.params, into vectors of size
        (n_cells +1). N.b. +1, as considers also the medium psuedo-cell.

        e.g. if self.params["A0"] = (5,6,7), then self.A0 will be 5, for cell-type 1, will be 6 for cell-type 2 etc.
        """
        self.A0 = np.zeros(self.n_cells + 1)
        self.P0 = np.zeros(self.n_cells + 1)
        self.lambda_A = np.zeros(self.n_cells + 1)
        self.lambda_P = np.zeros(self.n_cells + 1)
        for i, c_type in enumerate(self.c_types):
            self.A0[i + 1] = self.params["A0"][c_type - 1]
            self.P0[i + 1] = self.params["P0"][c_type - 1]
            self.lambda_A[i + 1] = self.params["lambda_A"][c_type - 1]
            self.lambda_P[i + 1] = self.params["lambda_P"][c_type - 1]
        self.make_J()

    def make_J(self):
        """
        In the case of homogeneous prescription of interfacial energies, utilise the "W" matrix.

        W matrix is a (n_cell_type + 1, n_cell_type + 1) of affinities (more positive = stronger affinity).

        This is derived from the self.params dictionary.

        self.J then is a (n_cell x n_cell) matrix of interfacial energy coefficients.
        (more negative = stronger affinity).

        Additionally calculates J_diff via the below function.
        @return:
        """
        self.J = np.zeros([self.n_cells + 1, self.n_cells + 1])
        c_types = np.concatenate(((0,), self.c_types))
        for i in range(len(c_types)):
            for j in range(len(c_types)):
                if i != j:
                    self.J[i, j] = -self.params["W"][c_types[i], c_types[j]]
        self.get_J_diff()

    def get_J_diff(self):
        """
        J_diff: Change in the interfacial energy when a pixel is replaced from cell index i to cell index j.
        Jdiff is a (nc x nc x nc) array, where the first two dimensions are indices of cells i and j, and the third
        dimension can be used to index of all neighbouring cells of the pixel that is being flipped.
        """
        self.J_diff = np.expand_dims(self.J, 1) - np.expand_dims(self.J, 0)

    def make_init(self, init_type="circle", r=3, spacing=0.25):
        """
        Initialise the I matrix with 'cells' (i.e. regions of the I matrix with a given cell index).

        For example, cells are prescribed as circles. This is achieved by defining a set of circle centres that are
        grid-tiled in the I matrix. These have the same radius, r. Cell indices and cell centres are shuffled, such that
        the initial distribution of cells is random. Extra cells are added, and then are removed from the centre of the
        I matrix outward until the appropriate number of cells, prescribed in **generate_cells**, is achieved.

        @param init_type: choose the type of initial cell shape. Currently, only circle.
        @param r: Radius of the circle that prescribes the contiguous region that a cell is initialised to.
        @param spacing: spacing between the circles.
        """
        self.I0 = np.zeros_like(self.I)
        if init_type == "circle":
            X, Y = np.meshgrid(np.arange(self.num_x), np.arange(self.num_y), indexing="ij")
            sq_n_x = int(np.ceil(np.sqrt(self.n_cells))) - 1
            sq_n_y = (int(np.ceil(np.sqrt(self.n_cells)))) / 2
            x_mid, y_mid = int(self.num_x / 2), int(self.num_y / 2)
            grid_spacing_x = np.arange(-sq_n_x, sq_n_x + 1) * (r + spacing)
            grid_spacing_y = np.arange(-sq_n_y, sq_n_y + 1) * (r + spacing) * np.sqrt(2)
            x0s, y0s = x_mid + grid_spacing_x, y_mid + grid_spacing_y
            X0, Y0 = np.meshgrid(x0s, y0s, indexing="ij")
            X0, Y0 = np.concatenate([X0[::2, ::2].ravel(), X0[1::2, 1::2].ravel()]), np.concatenate(
                [Y0[::2, ::2].ravel(), Y0[1::2, 1::2].ravel()])
            X0 += np.random.uniform(0, 0.01, X0.shape)
            Y0 += np.random.uniform(0, 0.01, Y0.shape)
            dist_to_mid = (X0 - x_mid) ** 2 + (X0 - y_mid) ** 2
            grid_choice = np.argsort(dist_to_mid)
            k = 0
            cell_index = np.arange(self.n_cells)
            random.shuffle(cell_index)
            while k < self.n_cells:
                x0, y0 = X0[grid_choice[k]], Y0[grid_choice[k]]
                cll_r = np.sqrt(self.A0[k + 1] / np.pi) * 0.8
                self.I0[(X - x0 + 0.5) ** 2 + (Y - y0 + 0.5) ** 2 <= cll_r ** 2] = cell_index[k] + 1
                k += 1
        self.I = self.I0.copy()
        self.assign_AP()

    def assign_AP(self):
        """
        Calculate the area and perimeter of every cell in the matrix, I.

        self.A and self.P are (n_cells + 1) vectors of area and perimeter values, respectively. These are indexed via
        the indices of cells in I. The first value is that of the medium, which is essentially ignored throughout the
        base, as we do not consider the area and perimeter of the medium psuedo-cell in the energy functional.
        """
        self.A = np.zeros(self.n_cells + 1, dtype=int)
        self.P = np.zeros(self.n_cells + 1, dtype=int)
        for cll in self.cell_ids:
            self.P[cll], self.A[cll] = self.get_perimeter_and_area(self.I, cll)

    def get_perimeter_and_area(self, I, s):
        """
        Calculates the area and perimeter of a given cell id, s, given the matrix of pixels, I.
        @param I: The (num_x x num_y) matrix of ints. Each int is the index corresponds to a cell. Thus each index is Moore contiguous within I.
        @param s: Index of the pixel in question.
        """
        M = I == s
        PI = np.sum(np.array([M != np.roll(np.roll(M, i, axis=0), j, axis=1) for i, j in self.perim_neighbour]), axis=0)
        P = np.sum(PI * M)
        A = np.sum(M)
        return P, A

    def initialize(self, J0, n_initialise_steps=10000):
        """
        Initialise the I matrix by performing M-H steps, after defining the approximate initialisation in **make_init**.

        See self.simulate function for more details, as these two functions are very analogous.

        @param J0: Definition of the J-matrix for the initialisation steps.
        @param n_initialise_steps: Number of initialisation steps.
        """
        J = self.J.copy()
        lambda_P = self.lambda_P.copy()
        self.lambda_P[:] = np.max(self.lambda_P)
        self.J = np.zeros_like(self.J)
        self.J[1:, 1:] = J0
        self.J = self.J * (1 - np.eye(self.J.shape[0]))
        self.get_J_diff()
        self.sample.n_steps = n_initialise_steps
        self.sample.do_steps()
        self.J = J.copy()
        self.lambda_P = lambda_P.copy()
        self.get_J_diff()
        print("initialized")

    def simulate(self, n_step, n_save, initialize=True, J0=None, n_initialise_steps=10000):
        """
        Simulate the CPM algorithm.

        Relies heavily on the **Sample** class in the **sample** module.

        Given an uninitialised I matrix, first initialise it with self.initialise, provided initialize is True.

        Then perform the M-H algorithm for **n_step**, under the energy functional, with parameters prescribed in
        self.params.

        Save this every m steps, such that n_save snapshots are saved at the end. This reduces the memory overhead of
        the function.

        Prints the percentage of the simulation that is complete every time that a snapshot is saved.

        @param n_step: Total number of iterations of the M-H algorithm (not including initialization steps)
        @param n_save: Number of snapshots saved for further analysis.
        @param initialize: Boolean. If True, then initialise under the M-H algorithm with self.initialize.
        @param J0: Definition of the J-matrix for the initialisation steps.
        @param n_initialise_steps: Number of initialisation steps.
        """
        if initialize:
            self.initialize(J0, n_initialise_steps)
        self.n_step = n_step
        self.skip = int(n_step / n_save)
        self.sample.n_steps = self.skip
        self.t = np.arange(n_step)
        self.t_save = self.t[::self.skip]
        self.n_save = len(self.t_save)
        self.I_save = np.zeros((self.n_save + 1, self.num_x, self.num_y), dtype=int)
        n_steps = int(n_step / self.skip)
        self.I_save[0] = self.I.copy()
        for i in range(n_steps):
            self.sample.do_steps()
            self.I_save[i + 1] = self.I.copy()
            print("%.1f" % (100 * (i / n_steps)))

    def save_simulation(self, dir_path, name):
        """
        Save the simulation to file. This is the set of I matrices in the snapshots of the self.simulate function
        i.e. of length n_save.

        This heavily compresses the data. Each I matrix is first made a csr sparse matrix (scipy.sparse module).
        As a list of sparse-matrices, then generate a compressed (pbz2) pickle file. Save to file.

        @param dir_path: Directory path into which the file will be saved.
        @param name: Name of the file (not including the extension, which is added).
        """
        self.I_save_sparse = [None] * len(self.I_save)
        for i, I in enumerate(self.I_save):
            self.I_save_sparse[i] = sparse.csr_matrix(I)
        with bz2.BZ2File(dir_path + "/" + name + '.pbz2', 'w') as f:
            cPickle.dump(self.I_save_sparse, f)

    def generate_image(self, I, res=8, col_dict={"E": "red", "T": "blue", "X": "green"},
                       background=np.array([0, 0, 0, 0.6])):
        """
        Genearte an image of the I matrix, colouring cells by cell-type, and establishing boundaries between cells
        such that cell outlines are visible.

        The I matrix is first scaled up by a multiplier **res**. I.e. if I was (100x100) and res is 8, then I_scale is
        (800 x 800).

        Then colour each cell_type under the col_dict.

        Find the perimeter elements, i.e. pixels that are on the boundary of cells. Colour these white.

        Then colour the non-boundary elements and non-cell elements with the colour background (RGBA format).

        This gives Im, a (num_x * res, num_y x res, 4) array of RBGA colours.

        @param I: The (num_x x num_y) matrix of ints. Each int is the index corresponds to a cell. Thus each index is
        Moore contiguous within I.
        @param res: resolution multiplier, defining the size of the final image. In essence, this defines how narrow the
        boundaries of cells are with respect to their areas.
        @param col_dict: Dictionary of colours for each of the cell types.
        @param background: RGBA colour of the background/medium.
        @return:
        """
        I_scale = np.repeat(np.repeat(I, res, axis=0), res, axis=1)
        Im = np.zeros([I.shape[0] * res, I.shape[1] * res, 4]).astype(float)
        Im[:, :, :] = background

        for j in range(1, self.n_cells + 1):
            cll_mask = I_scale == j
            cll_type = self.c_types[j - 1]
            col_name = col_dict.get(cll_type)
            if type(col_name) is str:
                col = np.array(colors.to_rgba(col_name))
            else:
                col = col_name
            Im[cll_mask] = col
        boundaries = self.get_perimeter_elements(I_scale)
        I_scale[boundaries] = 0
        Im[boundaries] = 0
        return Im

    def get_perimeter_elements(self, I):
        """
        Get a mask of all pixels that are adjacent to pixels of different indices (i.e. the cell boundaries).
        @param I: The (num_x x num_y) matrix of ints. Each int is the index corresponds to a cell. Thus each index is
        Moore contiguous within I.
        @return:
        """
        PI = np.sum(np.array([I != np.roll(np.roll(I, i, axis=0), j, axis=1) for i, j in self.neighbour_options]),
                    axis=0)
        return (PI != 0)

    def generate_image_t(self, res=8, col_dict={"E": "red", "T": "blue"}, background=np.array([0, 0, 0, 0.6])):
        """
        Iterate **self.generate_image** over all of the snapshots in I_save (of length n_save).

        Defines self.Im_save, a (n_save x num_x * res, num_y x res, 4) array of RBGA colours
        @param res: resolution multiplier, defining the size of the final image. In essence, this defines how narrow the
        boundaries of cells are with respect to their areas.
        @param col_dict: Dictionary of colours for each of the cell types.
        @param background: RGBA colour of the background/medium.
        """
        n_save = self.I_save.shape[0]
        Im_save = np.zeros([n_save, self.I_save.shape[1] * res, self.I_save.shape[2] * res, 4])
        for ni, I in enumerate(self.I_save):
            Im_save[ni] = self.generate_image(I, res, col_dict, background)
        self.Im_save = Im_save

    def animate(self, file_name=None, dir_name="plots"):
        """
        Generate an mp4 animation of the CPM simulation output. Uses generate_image_t and the ffmpeg encoder.
        @param file_name: Name of the animation file. If None, then will use the time-stamp.
        @param dir_name: Directory into which the animation file will be saved.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        def animate(i):
            ax1.clear()
            ax1.set(aspect=1)
            ax1.axis('off')
            ax1.imshow(self.Im_save[i])

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=1800)

        if file_name is None:
            file_name = "animation %d" % time.time()

        an = animation.FuncAnimation(fig, animate, frames=self.I_save.shape[0], interval=200)
        an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)

