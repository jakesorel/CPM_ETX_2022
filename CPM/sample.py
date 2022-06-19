#!/usr/bin/env python3

"""
This module takes in a *CPM* object, containing information about tissue geometry,
and evolves it under the Cellular-Potts/Metropolis Hastings algorithm to stochastically minimize the energy functional.

"""

import numpy as np
from numba import jit

from zmasks import Zmasks


class Sample:
    """
    **Sample** class, wrapping functions that perform the Metropolis-Hastings sampling algorithm.
    """

    def __init__(self, cpm, n_steps=100):
        """
        Initialise **Sample** class.
        @param cpm: a CPM object, on which the Metropolis-Hastings optimisation algorithm is performed.
        See corresponding documentation
        @param n_steps: number of steps to perform, every time the **do_steps** function is called.
        """
        self.cpm = cpm  # a CPM object.
        self.zmasks = Zmasks()  # initialise the **Zmasks** object. See corresponding documentation.
        self.T = self.cpm.params["T"]  # alias for the temperature of the optimisation. Prescribed in the **CPM** class.
        self.n_steps = n_steps  # number of steps to perform, every time the **do_steps** function is called.

    def do_step(self):
        """
        Wrapper for the **do_step** function, see below.

        Performs the function, then transfers the results back to the CPM class.
        @return:
        """
        self.cpm.I, self.cpm.A, self.cpm.P = do_step(self.cpm.I, self.cpm.num_x, self.cpm.num_y, self.zmasks.dP_z,
                                                     self.cpm.A, self.cpm.P, self.cpm.lambda_A, self.cpm.lambda_P,
                                                     self.cpm.A0, self.cpm.P0, self.cpm.J_diff, self.T,
                                                     self.zmasks.primes, self.zmasks.hashes)

    def do_steps(self):
        """
        Wrapper for the **do_steps** function, see below.

        Performs the function, then transfers the results back to the CPM class.
        @return:
        """
        self.cpm.I, self.cpm.A, self.cpm.P = do_steps(self.n_steps, self.cpm.I, self.cpm.num_x, self.cpm.num_y,
                                                      self.zmasks.dP_z, self.cpm.A, self.cpm.P, self.cpm.lambda_A,
                                                      self.cpm.lambda_P, self.cpm.A0, self.cpm.P0, self.cpm.J_diff,
                                                      self.T, self.zmasks.primes, self.zmasks.hashes)


@jit(nopython=True)
def do_step(I, num_x, num_y, dP_z, A, P, lambda_A, lambda_P, A0, P0, J_diff, T, primes, hashes):
    """
    Performs one iteration of the Metropolis-Hastings algorithm.

    @param I: The (num_x x num_y) matrix of ints. Each int is the index corresponds to a cell. Thus each index is Moore contiguous within I.
    @param num_x: Number of pixels in the x-dimension of I.
    @param num_y: Number of pixels in the y-dimension of I.
    @param dP_z: The change of perimeter given a specific type of swap. Indexed with respect to the z-mask list (see documentation in the **zmasks** module).
    @param A: Vector of areas, indexed with respect to cell indices prescribed in I.
    @param P: Vector of cell perimeters, indexed with respect to cell indices prescribed in I.
    @param lambda_A: The coefficient for the (A-A0) term in the energy functional. Cell-wise.
    @param lambda_P: The coefficient for the (P-P0) term in the energy functional. Cell wise.
    @param A0: Optimal area for each cell. Cell wise.
    @param P0: Optimal perimeter for each cell. Cell wise.
    @param J_diff: Change in the interfacial energy when a pixel is replaced from cell index i to cell index j.
    Jdiff is a (nc x nc x nc) array, where the first two dimensions are indices of cells i and j, and the third
    dimension can be used to index of all neighbouring cells of the pixel that is being flipped.
    @param T: Psuedo-temperature, used in the Metropolis-Hastings algorithm.
    @param primes: The kernel used to hash the local Moore neighbourhood.
    @param hashes: The set of hashes for all accepted local Moore neighbourhoods.
    @return: I_new, the new I matrix. A, the new areas. P, the new perimeters.
    """
    ##Given an existing I matrix, sample a random point, and then select the state of one of its Neumann neighbours.
    # (i,j) is the coordinate in the I matrix of the selected pixel.
    # s is the cell index of the chosen pixel
    # s2 is the cell index of the chosen neighbour
    i, j, s, s2 = pick_pixel(I, num_x, num_y)

    # Initialise the changes in the Energy/Hamiltonian as 0.
    dH_1, dH_2 = 0, 0

    # Given a point (i,j), subset I to find the Moore neighbourhood. Na is a matrix of size (3,3)
    Na = get_Na(I, i, j)

    # Initialise mask_ids.
    mask_id_1, mask_id_2 = 0, 0

    # Initialise the contributions of the change in the energy.
    dP_1 = 0
    dP_2 = 0
    dA_1 = 0
    dA_2 = 0

    # Calculate the change in energy. Calculated as two components: changes in state s and changes in state s2.
    if s != 0:  # if the chosen pixel is a medium cell.

        # The mask Na==s generically defines the neighbourhood. Only certain masks are allowed in order to preserve
        # local Moore contiguity and hence global Moore contiguity.

        # This is achieved by indexing the hashes of the allowed masks.
        # mask_id is the index of the mask Na==s within the pre-defined list of acceptable masks.
        mask_id_1 = get_mask_id(Na == s, primes, hashes)
        if mask_id_1 != -1:  # if Na==s is in the list of acceptable masks, then calculate the changes in
            # area and perimeter and hence the energy change.
            dP_1 = dP_z[mask_id_1]
            dA_1 = -1
            dH_1 = get_dH(s, dP_1, dA_1, A, P, lambda_A, lambda_P, A0, P0)
    if s2 != 0:  # if the chosen neighbour state is a medium cell.
        # Likewise with above, but instead with the swapped pixel.
        mask_id_2 = get_mask_id(Na == s2, primes, hashes)
        if mask_id_2 != -1:
            dP_2 = dP_z[mask_id_2]
            dA_2 = 1
            dH_2 = get_dH(s2, dP_2, dA_2, A, P, lambda_A, lambda_P, A0, P0)
    # Calculate the change in the interfacial energy term.
    dJ = get_dJ(J_diff, s, s2, Na)

    # Sum together the changes in the contributions of the energy to calculate the total change in energy: dH.
    dH = dH_1 + dH_2 + dJ

    # Copy I to putatively change I by swapping states. I2 may be accepted or rejected.
    I_new = I.copy()
    if (mask_id_1 != -1) * (mask_id_2 != -1):  # if both masks (Na==s) and (Na==s2) are permissible.
        if dH <= 0:  # if the change in energy is less than 0.
            # Swap the state of pixel (i,j) with the state of the neighbours.
            I_new[i, j] = s2
            # Update the properties of the cells.
            A[s] += dA_1
            A[s2] += dA_2
            P[s] += dP_1
            P[s2] += dP_2
        else:
            if np.random.random() < np.exp(-dH / T):  # stochastic contribution to minimisation, under M-H.
                # Swap the state of pixel (i,j) with the state of the neighbours.
                I_new[i, j] = s2
                # Update the properties of the cells.
                A[s] += dA_1
                A[s2] += dA_2
                P[s] += dP_1
                P[s2] += dP_2
    return I_new, A, P


@jit(nopython=True)
def do_steps(n_steps, I, num_x, num_y, dP_z, A, P, lambda_A, lambda_P, A0, P0, J_diff, T, primes, hashes):
    """
    Iterate **do_step** for n_steps.

    @param n_steps: Number of iteration steps.
    @param I: The (num_x x num_y) matrix of ints. Each int is the index corresponds to a cell. Thus each index is Moore contiguous within I.
    @param num_x: Number of pixels in the x-dimension of I.
    @param num_y: Number of pixels in the y-dimension of I.
    @param dP_z: The change of perimeter given a specific type of swap. Indexed with respect to the z-mask list (see documentation in the **zmasks** module).
    @param A: Vector of areas, indexed with respect to cell indices prescribed in I.
    @param P: Vector of cell perimeters, indexed with respect to cell indices prescribed in I.
    @param lambda_A: The coefficient for the (A-A0) term in the energy functional. Cell-wise.
    @param lambda_P: The coefficient for the (P-P0) term in the energy functional. Cell wise.
    @param A0: Optimal area for each cell. Cell wise.
    @param P0: Optimal perimeter for each cell. Cell wise.
    @param J_diff: Change in the interfacial energy when a pixel is replaced from cell index i to cell index j.
    Jdiff is a (nc x nc x nc) array, where the first two dimensions are indices of cells i and j, and the third
    dimension can be used to index of all neighbouring cells of the pixel that is being flipped.
    @param T: Psuedo-temperature, used in the Metropolis-Hastings algorithm.
    @param primes: The kernel used to hash the local Moore neighbourhood.
    @param hashes: The set of hashes for all accepted local Moore neighbourhoods.
    @return: I, the new I matrix. A, the new areas. P, the new perimeters.
    """
    for i in range(n_steps):
        I, A, P = do_step(I, num_x, num_y, dP_z, A, P, lambda_A, lambda_P, A0, P0, J_diff, T, primes, hashes)
    return I, A, P


@jit(nopython=True)
def H(A, P, lambda_A, lambda_P, A0, P0):
    """
    Calculate the energy of a given cell.
    @param A: Area of the cell.
    @param P: Its perimeter.
    @param lambda_A: The coefficient for the (A-A0) term in the energy functional. Cell-wise.
    @param lambda_P: The coefficient for the (P-P0) term in the energy functional. Cell wise.
    @param A0: Optimal area for each cell. Cell wise.
    @param P0: Optimal perimeter for each cell. Cell wise.
    @return: H, the energy of the cell inputted.
    """
    return lambda_A * (A - A0) ** 2 + lambda_P * (P - P0) ** 2


@jit(nopython=True)
def get_dH(s, dP, dA, A, P, lambda_A, lambda_P, A0, P0):
    """
    Calculate the change in energy.

    @param s: Index of the pixel in question.
    @param dP: Change in perimeter of that cell.
    @param dA: Change in area.
    @param A: Vector of areas, indexed with respect to cell indices prescribed in I.
    @param P: Vector of cell perimeters, indexed with respect to cell indices prescribed in I.
    @param lambda_A: The coefficient for the (A-A0) term in the energy functional. Cell-wise.
    @param lambda_P: The coefficient for the (P-P0) term in the energy functional. Cell wise.
    @param A0: Optimal area for each cell. Cell wise.
    @param P0: Optimal perimeter for each cell. Cell wise.
    @return:
    """
    dH = H(A[s] + dA, P[s] + dP, lambda_A[s], lambda_P[s], A0[s], P0[s])
    dH -= H(A[s], P[s], lambda_A[s], lambda_P[s], A0[s], P0[s])
    return dH


@jit(nopython=True)
def pick_pixel(I, num_x, num_y):
    """
    Algorithm to choose pixels.

    1. Randomly choose a pixel (i,j)
    2. Continue if: not a boundary pixel, else return to 1.
    3. Define the cell index of the pixel (i,j) as s
    4. Pick one of the indices of the neighbouring pixels (Neumann). s2.
    5. Accept if s=/=s2. Else return to 1.

    @param I: The (num_x x num_y) matrix of ints. Each int is the index corresponds to a cell. Thus each index is Moore contiguous within I.
    @param num_x: Number of pixels in the x-dimension of I.
    @param num_y: Number of pixels in the y-dimension of I.
    @return: i: Chosen pixel x-component.
    j: Chosen pixel y-component.
    s: Index of the pixel in question.
    s2: Index of the neighbouring.
    """
    picked = False
    while picked is False:
        i = int(np.random.random() * num_x)
        j = int(np.random.random() * num_y)
        if not ((i * j == 0) or (((i - num_x + 1) * (j - num_y + 1)) == 0)):
            s = I[i, j]
            s2 = get_s2(I, i, j, num_x, num_y)
            if s != s2:
                picked = True
    return i, j, s, s2


@jit(nopython=True)
def get_mask_id(the_mask, primes, hashes):
    """
    The mask Na==s generically defines the neighbourhood. Only certain masks are allowed in order to preserve
    local Moore contiguity and hence global Moore contiguity.

    This is achieved by indexing the hashes of the allowed masks.
    mask_id is the index of the mask Na==s within the pre-defined list of acceptable masks.


    @param the_mask: A 3x3 boolean array defining the neighbourhood of a pixel. In the code this is Na==s or Na==s2.
    @param primes: The kernel used to hash the mask.
    @param hashes: The list of acceptable hashes.
    @return: This function returns the index of the mask from the list of acceptable masks.
    If **the_mask** is not in this list, -1 is returned.
    """
    hash = np.sum(the_mask * primes)
    cont = True
    k = 0
    mask_id = -1
    n_hashes = len(hashes)
    while (cont) and (k < n_hashes):
        if hash == hashes[k]:
            mask_id = k
            cont = False
        else:
            k += 1
    return mask_id


@jit(nopython=True)
def get_Na(I, i, j):
    """
    Given a point (i,j), subset I to find the Moore neighbourhood. Na is a matrix of size (3,3)
    @param I: The (num_x x num_y) matrix of ints. Each int is the index corresponds to a cell. Thus each index is Moore contiguous within I.
    @param i: Chosen pixel x-component.
    @param j: Chosen pixel y-component.
    @return: Na. The 3x3 subset I centred on (i,j).
    """
    Na = I[i - 1:i + 2, j - 1:j + 2]
    return Na


@jit(nopython=True)
def get_s2(I, i, j, num_x, num_y):
    """
    Given a pixel in I, (i,j), randomly sample a cell index, s2, from the Neumann neighbourhood.
    Neumann neighbourhood is defined in neighbour_options.

    @param I: The (num_x x num_y) matrix of ints. Each int is the index corresponds to a cell. Thus each index is Moore contiguous within I.
    @param i: Chosen pixel x-component.
    @param j: Chosen pixel y-component.
    @param num_x: Number of pixels in the x-dimension of I.
    @param num_y: Number of pixels in the y-dimension of I.
    @return: s2, the cell index of the neighbouring pixel that is sampled randomly.
    """
    # Specifies Neumann neighbourhood.
    neighbour_options = np.array([[1, 0],
                                  [-1, 0],
                                  [0, 1],
                                  [0, -1]])

    # Randomly choose one of the four options. This defines the shift in the x and y directions wrt. (i,j).
    ni = neighbour_options[int(np.random.random() * 4)]

    # Identify s2 from I.
    s2 = I[np.mod(ni[0] + i, num_x), np.mod(ni[1] + j, num_y)]
    return s2


@jit(nopython=True)
def get_dJ(J_diff, s, s2, Na):
    """
    Calculate the change in the interfacial energy, dJ.

    @param J_diff: Change in the interfacial energy when a pixel is replaced from cell index i to cell index j.
    @param s: Index of the pixel in question.
    @param s2: Index of the neighbouring pixel.
    @param Na: A 3x3 matrix, subsetting I centred on (i,j).
    @return:
    """
    Js2s = J_diff[s2, s]
    dJ = Js2s.take(Na.take([0, 1, 2, 3, 5, 6, 7, 8])).sum()
    return dJ
