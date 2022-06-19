#!/usr/bin/env python3

"""
A script for running a suite of analyses on CPM simulation files.

Run from the command-line:

e.g. python run_analysis.py 72

where 72 defines the bootstrap adhesion matrix that is to be used for parameterising the CPM, and hence the simulation
output.
"""

import _pickle as cPickle
import bz2
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage.measurements import center_of_mass

if __name__ == "__main__":

    #Redefine locally the cell_types.
    c_types = np.zeros(23, dtype=int)
    c_types[1:] = 1
    c_types[9:] = 2
    c_types[17:] = 3

    #Booleans of whether each of the cell_type indices are ES, TS or XEN.
    E_mask = c_types == 1
    T_mask = c_types == 2
    X_mask = c_types == 3

    #Define the neighbourhood.
    i_, j_ = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
    Moore = np.array([i_.ravel(), j_.ravel()]).T
    i_2, j_2 = np.delete(i_.ravel(), 4), np.delete(j_.ravel(), 4)
    perim_neighbour = np.array([i_2, j_2]).T
    perim_neighbour_reduced = np.array([[0, 1], [1, 0], [1, 1]])

    #The index of the simulation/bootstrap-adhesion matrix.
    iter_i = int(sys.argv[1])


    def get_adj(I_sparse):
        """
        Calculate the adjacency matrix between each of the cells.

        :param I_sparse: csr-sparse (scipy.sparse) format of cell-indices, i.e. the I matrix.
        :return:
        """
        I = I_sparse.toarray()
        vals = []
        neighs = []
        for i, j in perim_neighbour_reduced:
            rolled = np.roll(np.roll(I, i, axis=0), j, axis=1)
            mask = I != rolled
            vals += list(I[mask])
            neighs += list(rolled[mask])
        adj = sparse.csr_matrix(
            sparse.coo_matrix(([True] * len(vals), (vals, neighs)), shape=(len(c_types), len(c_types))))
        adj += adj.T
        return adj


    def get_conn_comp(adj, c_types_i):
        """
        Calcylate the number of connected components of each cell-type, given the adjacency matrix (see above).

        :param adj: adjacency matrix, defined by **get_adj**
        :param c_types_i: cell_type index vector, which has been stripped of non-attached cells
        (see **remove_non_attached**).
        :return: Number of connected components for ES, TS and XEN (in this order)
        """
        conn_comp = np.zeros(3, dtype=int)
        for i in range(1, 4):
            mask = c_types_i == i
            conn_comp[i-1] = sparse.csgraph.connected_components(adj[mask].T[mask].T)[0]
        return conn_comp


    def remove_non_attached(I_sparse_):
        """
        For a given I matrix (in sparse format, I_sparse_), find the cells that have only 'medium' neighbours and are
        hence non-attached to the main aggregate and remove them.

        Strictly, the cells that remain after this removal are those in the largest connected component in the adjacency
        matrix.
        :param I_sparse_: csr-sparse (scipy.sparse) format of cell-indices, i.e. the I matrix.
        :return: I_sparse, (I matrix, where all non-attached cells are replaced by medium = 0), and c_types_i, the
        cell_type vector for the remaining cells.
        """
        I_sparse = I_sparse_.copy()
        adj = get_adj(I_sparse)
        n_cc, cc_id = sparse.csgraph.connected_components(adj[1:, 1:])
        c_types_i = c_types.copy()
        if n_cc > 1:
            dom_id = np.argmax(np.bincount(cc_id))
            mask = cc_id != dom_id
            cids_drop = np.nonzero(mask)[0]
            for cid in cids_drop:
                I_sparse[I_sparse == cid + 1] = 0
                c_types_i[cid + 1] = 0
        return I_sparse, c_types_i


    def get_n_external(adj, c_types_i):
        """
        One metric to calculate the number of external cells.

        A cell is external if it is a direct neighbour of a medium cell.
        :param adj: adjacency matrix, defined by **get_adj**
        :param c_types_i: cell_type index vector, which has been stripped of non-attached cells
        (see **remove_non_attached**).
        :return: Number of external cells (by cell-type)
        """
        i, j = np.nonzero(adj)
        j_out = j[i == 0]
        return np.bincount(c_types_i.take(j_out), minlength=4)[1:]


    def get_n_external_2(adj, c_types_i):
        """
        The most parsimonous external metric, used in the paper Bao et al., 2022 Nature Cell Biology.

        A cell is external if it is found in a connected component of that cell type, where at least one of the cells
        flanks the medium.

        :param adj: adjacency matrix, defined by **get_adj**
        :param c_types_i: cell_type index vector, which has been stripped of non-attached cells
        (see **remove_non_attached**).
        :return: Number of external cells (by cell-type)
        """
        n_ext = np.zeros(3, dtype=int)
        b_mask = np.zeros_like(E_mask)
        b_mask[0] = True
        for i in range(3):
            msk = c_types_i == (i + 1)
            mask = msk + b_mask
            bc = np.bincount(sparse.csgraph.connected_components(adj[mask].T[mask].T)[1][1:])
            ne = sum(msk)
            if bc.size > 1:
                ne -= bc[1:].sum()
            n_ext[i] = ne
        return n_ext


    def get_most_external_from_unrolled(unrolled):
        """
        Given a radial projection of I ('unrolled'), find the cell that is most external, for each theta value.

        See **get_n_external_3** for reference.

        :param unrolled: Radial projection of I.
        :return: Most external cell index.
        """
        nr = unrolled.shape[1]
        nt = unrolled.shape[0]
        most_external = np.zeros(nt, dtype=int)
        for j in range(nt):
            me = 0
            k = 0
            while (k < nr) * (me == 0):
                me = unrolled[j, -k - 1]
                if me == 0:
                    k += 1
            most_external[j] = me
        return most_external


    def get_n_external_3(I_sparse, c_types_i):
        """
        A metric to determine the number of cells of each type that are 'external'. In the paper, we went for a more
        direct topological metric, meaning this is not used.

        The I matrix (image) undergoes a radial re-projection. Specifically, the centre of mass of (non-medium) cells is
        calculated and used as the centroid of the tissue. Then the image is sampled at variable (theta,r) wrt. to
        this centre of mass to establish a radial projection.


        :param I_sparse_: csr-sparse (scipy.sparse) format of cell-indices, i.e. the I matrix.
        :param c_types_i: cell_type index vector, which has been stripped of non-attached cells
        (see **remove_non_attached**).
        :return:
        """
        I = I_sparse.toarray()
        mid_pt = np.array(center_of_mass(I != 0))
        r_max = (100 - mid_pt).min()
        r_max = np.min((r_max, np.min(mid_pt)))
        r_max *= 0.95
        r = np.linspace(0, r_max, 30)
        theta = np.expand_dims(np.linspace(0, np.pi * 2, 100), 1)
        x_i, y_i = mid_pt[0] + r * np.cos(theta), mid_pt[1] + r * np.sin(theta)
        x_i = np.round(x_i).astype(int)
        y_i = np.round(y_i).astype(int)
        unrolled = I[(x_i, y_i)]
        most_external = get_most_external_from_unrolled(unrolled)
        cll_in_theta = np.zeros((len(unrolled), len(c_types)), dtype=bool)
        for i, ct in enumerate(c_types_i):
            cll_in_theta[:, i] = (unrolled == i).any(axis=1)
        cll_same_ctype_as_external = (np.expand_dims(c_types_i.take(most_external), 1) == c_types_i)
        frac_external = (cll_in_theta * cll_same_ctype_as_external).sum(axis=0) / cll_in_theta.sum(axis=0)
        is_external = frac_external > 0.5
        ES_external = is_external[c_types_i == 1].sum()
        TS_external = is_external[c_types_i == 2].sum()
        XEN_external = is_external[c_types_i == 3].sum()
        return np.array([ES_external, TS_external, XEN_external])


    x_f,y_f = np.mgrid[:100,:100] + 0.5

    def enveloping_score(I_sparse,c_types_i):
        """
        Define whether a cell is enveloping or not.

        A cell is enveloping if its centre of mass lies outside of its perimeter.

        :param I_sparse_: csr-sparse (scipy.sparse) format of cell-indices, i.e. the I matrix.
        :param c_types_i: cell_type index vector, which has been stripped of non-attached cells
        (see **remove_non_attached**).
        :return: Boolean vector, True if enveloping, one per cell.
        """
        C = c_types_i.take(I_sparse.toarray())
        is_enveloping = np.zeros((3),dtype=bool)
        for i in range(3):
            j = i+1
            mid_pt = np.array(center_of_mass(C==j))
            mid_pt_coord = np.round(mid_pt).astype(int)
            mid_pt_type = C[mid_pt_coord[0],mid_pt_coord[1]]
            is_enveloping[i] = mid_pt_type!=j
        return is_enveloping

    def get_c_type_adj(adj,c_types_i):
        """
        Calculate the adjacency matrix of cell_types (i.e. (4 x 4) in our case, 3 cell-types + medium)

        :param adj: adjacency matrix, defined by **get_adj**
        :param c_types_i: cell_type index vector, which has been stripped of non-attached cells
        (see **remove_non_attached**).
        :return: Adjacency matrix of cell-types.
        """
        edges = adj.nonzero()
        c_edges = (c_types_i.take(edges[0]),c_types_i.take(edges[1]))
        adj_c = sparse.csr_matrix(sparse.coo_matrix(([True]*len(c_edges[0]),c_edges)))
        return adj_c

    def get_conformation_score(env_score_i,cc_i,adj,c_types_i):
        """
        Score an in-silico ETX as one of the 17 possible conformations.

        Scoring system:

        Class 1
        0 E-T-X:      (1,1,1),(0,0,0)

        Class 2
        1 X(E(T)):    (X,1,X),(1,-1,-1)
        2 T(E(X)):    (X,X,1),(-1,1,1),
        3 X(T(E)):    (1,X,X),(-1,-1,-1)
        4 E(T(X)):    (X,X,1),(1,1,1)
        5 T(X(E)):    (1,X,X),(-1,-1,1)
        6 E(X(T)):    (X,1,X),(1,1,-1)

        Class 3
        7 X(E-T):     (1,1,X),(X,-1,-1)
        8 T(E-X):     (1,X,1),(-1,X,1)
        9 E(T-X):     (X,1,1),(1,1,X)

        Class 4
        10 T(E)-X:     (1,X,1),(-1,X,X)
        11 X(E)-T:     (1,1,X),(X,-1,X)
        12 E(X)-T:     (X,1,1),(X,1,X)
        13 T(X)-E:     (1,X,1),(X,X,1)
        14 E(T)-X:     (X,1,1),(1,X,X)
        15 X(T)-E:     (1,1,X),(X,X,-1)

        16 unsorted:   if not any of above.


        If the centre of mass of a compartment lies within its own bounds, then it cannot be enveloping.
        Distinguishing between classes 3 and 4 (where two cell types are non-enveloping):
            Measure the number of cell types that each cell of the enveloping class is bound to (ignoring boundary).

            If the other two cell types are contiguous:
                If any cells don't flank a boundary:
                    Unsorted if any of these cells are surronded by exclusively one OTHER cell type
                    Class 4: if surrounded by exactly two OTHER cell types
                    Class 3: otherwise (i.e. surrounded by one or more cell types AND a similar cell).
                Else:
                    Class 4
            Else:
                Unsorted

            Distinguishing between Class 2:
                Consider (i) X(E(T)) vs (ii)  E(X(T))
                    If X flanks boundary, E and NOT T; E flanks T --> (i)
                    If E flanks boundary, X and not T; X flanks T --> (ii)
                    If neither --> unsorted

        :param env_score_i: Enveloping score, see above.
        :param cc_i: Connected components for each cell_type
        :param adj: adjacency matrix, defined by **get_adj**
        :param c_types_i: cell_type index vector, which has been stripped of non-attached cells
        (see **remove_non_attached**).
        :return: conformation score, see doc-string for descriptions.
        """
        n_env = np.sum(env_score_i)
        conformation = 16
        if n_env == 0:
            if np.all(cc_i == 1):
                conformation = 0
        elif n_env == 1:
            c_type_env = env_score_i@np.arange(1,4)
            if (cc_i[np.mod(c_type_env,3)] ==1 )*(cc_i[np.mod(c_type_env-2,3)]==1): #if the other two cell types are contiguous i.e. conn comp = 1
                adj_reduced = adj[c_types_i==c_type_env]
                c_types_flanking = np.zeros((adj_reduced.shape[0],4),dtype=bool)
                adj_reduced_nonzero = adj_reduced.nonzero()
                c_types_flanking[adj_reduced_nonzero[0],c_types_i.take(adj_reduced_nonzero)]=True
                c_types_flanking_excluding_env = c_types_flanking[:,np.arange(4)!=c_type_env]
                flanking_boundary =c_types_flanking[:,0]
                if flanking_boundary.all():
                    conformation = 6+c_type_env
                else:
                    n_flanking_types = c_types_flanking.sum(axis=1)
                    if (n_flanking_types[~flanking_boundary]!=1).all(): ##if there isn't a cell surrounded exclusively by a single cell type.
                        n_flanking_types_nonboundary_nonself = c_types_flanking_excluding_env[:, 1:].sum(axis=1)
                        if (n_flanking_types_nonboundary_nonself[~flanking_boundary]==2).any(): #if any of the non-boundary cells are surrounded by exactly two OTHER cell types
                            flanking_other_ctypes_count = c_types_flanking_excluding_env[:, 1:].sum(axis=0)
                            max_flanking_other = np.argmax(flanking_other_ctypes_count)
                            conformation = 10 + (c_type_env -1)*2 + max_flanking_other
                        else:
                            conformation = 6+c_type_env
        elif n_env == 2:
            c_type_3 = np.nonzero(~env_score_i)[0][0]+1
            if cc_i[c_type_3-1] == 1:
                c_type_env1, c_type_env2 = np.nonzero(env_score_i)[0]
                c_type_env1 += 1
                c_type_env2 += 1
                edges = adj.nonzero()
                c_type_adj = sparse.coo_matrix(([True]*len(edges[0]),(c_types_i.take(edges[0]),c_types_i.take(edges[1])))).todense()
                if c_type_adj[c_type_env1,0]*c_type_adj[c_type_env1,c_type_env2]*(~c_type_adj[c_type_env1,c_type_3])*c_type_adj[c_type_env2,c_type_3]:
                    conformation = 1 + (c_type_env1-1)*2 + int(c_type_3<c_type_env2)
                elif c_type_adj[c_type_env2,0]*c_type_adj[c_type_env2,c_type_env1]*(~c_type_adj[c_type_env2,c_type_3])*c_type_adj[c_type_env1,c_type_3]:
                    conformation = 1 + (c_type_env2 - 1) * 2 + int(c_type_3 < c_type_env1)

        return conformation



    def get_top_values_t(I_save_sparse):
        """
        Get topological values, across the entire simulation.

        Uses the above functions and applies them to each snapshot in I_save_sparse.

        :param I_save_sparse: a list of I_save sparse matrices. I_sparse_: csr-sparse (scipy.sparse) format of cell-indices, i.e. the I matrix.
        :return: tuple of topological values for each time-point.
        """
        conn_comp_t = np.zeros((len(I_save_sparse), 3), dtype=int)
        n_external_1_t = np.zeros((len(I_save_sparse), 3), dtype=int)
        n_external_2_t = np.zeros((len(I_save_sparse), 3), dtype=int)
        n_external_3_t = np.zeros((len(I_save_sparse), 3), dtype=int)
        n_ctypes = np.zeros((len(I_save_sparse), 3), dtype=int)
        external_direction_t = np.zeros((len(I_save_sparse), 3), dtype=int)
        env_score_t = np.zeros((len(I_save_sparse),3),dtype=bool)
        conformation_t = np.zeros((len(I_save_sparse)),dtype=int)
        for t, I_sparse_full in enumerate(I_save_sparse):
            I_sparse, c_types_i = remove_non_attached(I_sparse_full)
            n_ctypes[t] = np.bincount(c_types_i, minlength=4)[1:]
            adj = get_adj(I_sparse)
            conn_comp_t[t] = get_conn_comp(adj, c_types_i)
            n_external_1_t[t] = get_n_external(adj, c_types_i)
            n_external_2_t[t] = get_n_external_2(adj, c_types_i)
            n_external_3_t[t] = get_n_external_3(I_sparse, c_types_i)
            external_direction_t[t] = get_external_direction(adj, c_types_i, n_ctypes[t])
            env_score_t[t] = enveloping_score(I_sparse, c_types_i)
            conformation_t[t] = get_conformation_score(env_score_t[t], conn_comp_t[t], adj, c_types_i)
        return conn_comp_t, n_ctypes, n_external_1_t, n_external_2_t, n_external_3_t,external_direction_t,env_score_t,conformation_t


    def types_xy_n_external(ctype_i, ctype_j, c_types_i, adj):
        b_mask = np.zeros_like(E_mask)
        b_mask[(c_types_i != ctype_i) * (c_types_i != ctype_j)] = True

        adj_mod = adj.copy()

        i_mask = c_types_i == ctype_i
        j_mask = c_types_i == ctype_j
        k_mask = (c_types_i!=0)*(~i_mask)*(~j_mask)
        ij_mask = i_mask + j_mask
        adj_mod = sparse.csr_matrix(~(np.expand_dims(ij_mask,1)*(ij_mask))).multiply(adj_mod)
        k_flank_outside = np.zeros(adj_mod.shape,dtype=bool)
        k_flank_outside[0] = k_mask
        k_flank_outside[:,0] = k_mask
        adj_mod = adj_mod + sparse.csr_matrix(k_flank_outside)


        n_ext_i = np.bincount(sparse.csgraph.connected_components(adj_mod)[1][i_mask])[0]
        n_ext_j = np.bincount(sparse.csgraph.connected_components(adj_mod)[1][j_mask])[0]

        return n_ext_i,n_ext_j


    def get_n_external_pairs(adj, c_types_i):
        n_external_mat = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                if i>j:
                    n_external_mat[i, j],n_external_mat[j,i] = types_xy_n_external(i + 1, j + 1, c_types_i, adj)
        return n_external_mat

    def get_external_direction(adj, c_types_i,n_ctypes_i):
        n_external_mat = get_n_external_pairs(adj, c_types_i)
        external_direction = np.zeros(3,dtype=int)
        ##ET,EX,TX i.e. E external to T, E to X, T to X
        external_direction[0] = 1*(n_external_mat[0,1] == n_ctypes_i[0]) -1*(n_external_mat[1,0] == n_ctypes_i[1])
        external_direction[1] = 1*(n_external_mat[0,2] == n_ctypes_i[0]) -1*(n_external_mat[2,0] == n_ctypes_i[2])
        external_direction[2] = 1*(n_external_mat[1,2] == n_ctypes_i[1]) -1*(n_external_mat[2,1] == n_ctypes_i[2])
        return external_direction


    #Set up the directory architecture.
    if not os.path.exists("../results/compiled"):
        os.mkdir("../results/compiled")

    if not os.path.exists("../results/compiled/bootstrap"):
        os.mkdir("../results/compiled/bootstrap")

    if not os.path.exists("../results/compiled/scrambled"):
        os.mkdir("../results/compiled/scrambled")

    if not os.path.exists("../results/compiled/soft"):
        os.mkdir("../results/compiled/soft")

    if not os.path.exists("../results/compiled/stiff"):
        os.mkdir("../results/compiled/stiff")


    #Save the topology scores for each time-point, to a csv file.
    I_save_sparse = cPickle.load(bz2.BZ2File("../results/stiff/%d.pbz2" % iter_i, 'rb'))
    cc, n_ctypes, next, next2, next3,ext_dir,env_score,conformation = get_top_values_t(I_save_sparse)

    df = pd.DataFrame({"t": np.arange(cc.shape[0]) * 1e4,
                       "N_E": n_ctypes[:, 0], "N_T": n_ctypes[:, 1], "N_X": n_ctypes[:, 2],
                       "E_cc": cc[:, 0], "T_cc": cc[:, 1], "X_cc": cc[:, 2],
                       "E_ex": next[:, 0], "T_ex": next[:, 1], "X_ex": next[:, 2],
                       "E_ex2": next2[:, 0], "T_ex2": next2[:, 1], "X_ex2": next2[:, 2],
                       "E_ex3": next3[:, 0], "T_ex3": next3[:, 1], "X_ex3": next3[:, 2],
                       "ET_ext":ext_dir[:,0],"EX_ext":ext_dir[:,1],"TX_ext":ext_dir[:,2],
                       "E_enveloping":env_score[:,0],"T_enveloping":env_score[:,1],"X_enveloping":env_score[:,2],
                       "conformation":conformation})
    df.to_csv("../results/compiled/stiff/%d.csv" % iter_i)

    I_save_sparse = cPickle.load(bz2.BZ2File("../results/scrambled/%d.pbz2" % iter_i, 'rb'))

    cc, n_ctypes, next, next2, next3,ext_dir,env_score,conformation = get_top_values_t(I_save_sparse)

    df = pd.DataFrame({"t": np.arange(cc.shape[0]) * 1e4,
                       "N_E": n_ctypes[:, 0], "N_T": n_ctypes[:, 1], "N_X": n_ctypes[:, 2],
                       "E_cc": cc[:, 0], "T_cc": cc[:, 1], "X_cc": cc[:, 2],
                       "E_ex": next[:, 0], "T_ex": next[:, 1], "X_ex": next[:, 2],
                       "E_ex2": next2[:, 0], "T_ex2": next2[:, 1], "X_ex2": next2[:, 2],
                       "E_ex3": next3[:, 0], "T_ex3": next3[:, 1], "X_ex3": next3[:, 2],
                       "ET_ext":ext_dir[:,0],"EX_ext":ext_dir[:,1],"TX_ext":ext_dir[:,2],
                       "E_enveloping":env_score[:,0],"T_enveloping":env_score[:,1],"X_enveloping":env_score[:,2],
                       "conformation":conformation})
    df.to_csv("../results/compiled/scrambled/%d.csv" % iter_i)
