import json
import os
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from asap3.analysis import FullCNA
from ase.io import read
from mff import configurations, models
from mff.gp import GaussianProcess

# Keep track on whether there are actual energies in your dataset or they have been discarded
global energydefault
energydefault = False


def find_repulstion_sigma(confs):
    """ Function used to find the repulsion parameter 
    rep_sig such that the energy of a bond at distance r is 0.02 eV.
    The distance r is the smallest bond distance found in the training set.
    """

    dists = []
    for c in confs:
        if len(c.shape) == 2:
            d_ = np.sum(c[:, :3]**2, axis=1)**0.5
            dists.extend(d_)
        else:
            for c1 in c:
                d_ = np.sum(c1[:, :3]**2, axis=1)**0.5
                dists.extend(d_)

    r = min(dists)
    rep_sig = r*0.02**(1/12)

    return rep_sig


def get_repulsive_forces(confs, sig):
    """ Function used to get repulsive forces for a configuration
    given a sigma value. The repuslion is a LJ repulsion.
    """
    forces = np.zeros((len(confs), 3))
    for i, c in enumerate(confs):
        d_ = np.sum(c[:, :3]**2, axis=1)**0.5
        v = c[:, :3]/d_[:, None]
        f = 12*(sig/d_)**12/d_
        forces[i] = np.sum(f[:, None]*v, axis=0)
    return forces


def get_repulsive_energies(confs, sig, mapping=False):
    """ Function used to get repulsive energy for a configuration
    given a sigma value. The repuslion is a LJ repulsion.
    """
    energies = np.zeros(len(confs))
    if not mapping:
        for i, c in enumerate(confs):
            for c1 in c:
                d_ = np.sum(c1[:, :3]**2, axis=1)**0.5
                energies[i] += np.sum((sig/d_)**12)
    else:
        for i, c1 in enumerate(confs):
            d_ = np.sum(c1[:, :3]**2, axis=1)**0.5
            energies[i] += np.sum((sig/d_)**12)
    return energies


def open_data(folder, cutoff):
    """ Open already extracted conf, force and energy data
    """
    elements, confs, forces, energies, global_confs = configurations.load_and_unpack(
        folder, cutoff)
    print("Opened data from %s" % (folder))
    return elements, confs, forces, energies, global_confs


def extract_data(folder, cutoff, filename=None):
    """ Extract training points from an .xyz trajectory or an .out file
    """
    file_path = Path(folder + '/' + filename)
    data = configurations.generate_and_save(
        file_path, cutoff, forces_label='forces', energy_label='energy')
    elements, confs, forces, energies, global_confs = configurations.unpack(
        data)
    return elements, confs, forces, energies, global_confs


def get_data(folder, cutoff, filename=None):
    """ Retrieve the data either from a set of .npy objects or from an .xyz trajectory
    """

    try:
        elements, confs, forces, energies, global_confs = open_data(
            folder, cutoff)
        print("Loaded data from %s" % (folder))

    except FileNotFoundError:
        try:
            elements, confs, forces, energies, global_confs = extract_data(
                folder, cutoff, filename)
            print("Extracted data from %s" % (filename))
        except FileNotFoundError:
            sys.exit(
                "I did not found either the conf, force and energy files, or the movie.xyz file. Quitting now.")

    return elements, confs, forces, energies, global_confs


def get_manyfolders(folders, cutoff, train_filenames):
    """ Retrieve the data for many folders and join everything
    """

    confs, forces, energies, elements = [], [], [], []
    for f, name in zip(folders, train_filenames):
        a, b, c, d = get_data(f, cutoff, name)
        confs.extend(a)
        forces.extend(b)
        energies.extend(c)
        elements.extend(d)

    forces = np.asarray(forces)
    energies = np.asarray(energies)
    elements = list(np.unique(np.ravel(elements)))
    return confs, forces, energies, elements


def grid_2b_onesp(X, nbins, cutoff):
    """ Grid sampling for 2-body descriptor and system with one element
    """
    stored_histogram = np.zeros(nbins)
    index = []
    ind = np.arange(len(X))
    randomarange = np.random.choice(ind, size=len(X), replace=False)
    for j in randomarange:  # for every snapshot of the trajectory file
        distances = np.sqrt(np.einsum('id -> i', np.square(X[j][:, :3])))
        distances[np.where(distances > cutoff)] = None
        this_snapshot_histogram = np.histogram(distances, nbins, (0.0, cutoff))
        if (stored_histogram - this_snapshot_histogram[0] < 0).any():
            index.append(j)
            stored_histogram += this_snapshot_histogram[0]

    return index


def grid_2b_manysp(X, nbins, cutoff, elements):
    """ Grid sampling for 2-body descriptor and system with two elements
    """
    element_pairs = list(combinations_with_replacement(elements, 2))
    stored_histogram = np.zeros((nbins, len(element_pairs)))
    index = []
    ind = np.arange(len(X))
    randomarange = np.random.choice(ind, size=len(X), replace=False)

    for j in randomarange:  # for every snapshot of the trajectory file
        distances = np.sqrt(np.einsum('id -> i', np.square(X[j][:, :3])))
        distances[np.where(distances > cutoff)] = None
        for k in range(len(element_pairs)):
            if k == 1:  # In the case of two different elements, we have to account for permutation invariance
                this_element_pair = np.union1d(
                    np.intersect1d(np.where(X[j][:, 3] == element_pairs[k][0]),
                                   np.where(X[j][:, 4] == element_pairs[k][1])),
                    np.intersect1d(np.where(X[j][:, 3] == element_pairs[k][1]),
                                   np.where(X[j][:, 4] == element_pairs[k][0])))
            else:
                this_element_pair = np.intersect1d(
                    np.where(X[j][:, 3] == element_pairs[k][0]), np.where(X[j][:, 4] == element_pairs[k][1]))
            distances_this = distances[this_element_pair]

            this_snapshot_histogram = np.histogram(
                distances_this, nbins, (0.0, cutoff))
            if (stored_histogram[:, k] - this_snapshot_histogram[0] < 0).any():
                index.append(j)
                stored_histogram[:, k] += this_snapshot_histogram[0]

    return index


def grid_3b_onesp(X, nbins, cutoff):
    """ Grid sampling for 3-body descriptor and system with one element
    """
    stored_histogram = np.zeros((nbins, nbins, nbins))
    index = []
    ind = np.arange(len(X))
    randomarange = np.random.choice(ind, size=len(X), replace=False)
    for j in randomarange:  # for every snapshot of the trajectory file
        atoms = np.vstack(([0., 0., 0.], X[j][:, :3]))
        distances = cdist(atoms, atoms)
        distances[np.where(distances > cutoff)] = None
        distances[np.where(distances == 0)] = None
        triplets = []
        for k in np.argwhere(distances[:, 0] > 0):
            for l in np.argwhere(distances[0, :] > 0):
                if distances[k, l] > 0:
                    triplets.append(
                        [distances[0, k], distances[0, l], distances[k, l]])
                    triplets.append(
                        [distances[0, l], distances[k, l], distances[0, k]])
                    triplets.append(
                        [distances[k, l], distances[0, k], distances[0, l]])

        triplets = np.reshape(triplets, (len(triplets), 3))

        this_snapshot_histogram = np.histogramdd(triplets, bins=(nbins, nbins, nbins),
                                                 range=((0.0, cutoff), (0.0, cutoff), (0.0, cutoff)))

        if (stored_histogram - this_snapshot_histogram[0] < 0).any():
            index.append(j)
            stored_histogram += this_snapshot_histogram[0]

    return index


def grid_3b_manysp(X, nbins, cutoff, elements):
    """ Grid sampling for 3-body descriptor and system with two elements
    """
    possible_triplets = list(combinations_with_replacement(elements, 3))
    stored_histogram = np.zeros((nbins, nbins, nbins, len(possible_triplets)))
    index = []
    ind = np.arange(len(X))
    randomarange = np.random.choice(ind, size=len(X), replace=False)
    for j in randomarange:  # for every snapshot of the trajectory file
        atoms = np.vstack(([0., 0., 0.], X[j][:, :3]))
        distances = cdist(atoms, atoms)
        distances[np.where(distances > cutoff)] = None
        distances[np.where(distances == 0)] = None
        triplets = []
        elements_triplets = []
        for k in np.argwhere(distances[:, 0] > 0):
            for l in np.argwhere(distances[0, :] > 0):
                if distances[k, l] > 0:
                    elements_triplets.append(
                        np.sort([X[j][0, 3], X[j][k-1, 4], X[j][l-1, 4]]))
                    triplets.append(
                        [distances[0, k], distances[0, l], distances[k, l]])
                    triplets.append(
                        [distances[0, l], distances[k, l], distances[0, k]])
                    triplets.append(
                        [distances[k, l], distances[0, k], distances[0, l]])

        elements_triplets = np.reshape(
            elements_triplets, (len(elements_triplets), 3))
        triplets = np.reshape(triplets, (len(triplets), 3))
        this_snapshot_histogram = np.histogramdd(triplets, bins=(nbins, nbins, nbins),
                                                 range=((0.0, cutoff), (0.0, cutoff), (0.0, cutoff)))
        for k in np.arange(len(possible_triplets)):
            valid_triplets = triplets[np.where(
                elements_triplets == possible_triplets[k]), :][0]
            this_snapshot_histogram = np.histogramdd(valid_triplets, bins=(nbins, nbins, nbins),
                                                     range=((0.0, cutoff), (0.0, cutoff), (0.0, cutoff)))

            if (stored_histogram[:, :, :, k] - this_snapshot_histogram[0] < 0).any():
                index.append(j)
                stored_histogram[:, :, :, k] += this_snapshot_histogram[0]

    return index


def sample_oneset(c, f, gc, en, el, method, ntr, ntest, cutoff, nbins=None, f_e_ratio=100, traj=None, cna_cut=None):
    """ Get training and test set from one database with method of choice
    """
    # For the forces, isolate a test set at random and then apply database selection on the remaining data
    ind = np.arange(len(c))
    ind_test = np.random.choice(ind, size=ntest, replace=False)
    ind_train = np.array(list(set(ind) - set(ind_test)))

    # For the energy we always use random sampling
    ntr_e = ntr//f_e_ratio + 1
    ntest_e = ntest//f_e_ratio+1
    ind_e = np.random.choice(np.arange(len(gc)), ntr_e+ntest_e, replace=False)
    ind_train_e = ind_e[ntr_e:]
    ind_test_e = ind_e[:ntr_e]

    if ((en == None).any()):
        en = np.zeros(len(c))
        energydefault = True

    X, Y, X_e, Y_e = c[ind_train], f[ind_train], gc[ind_train_e], en[ind_train_e]
    x, y, x_e, y_e = c[ind_test], f[ind_test], gc[ind_test_e], en[ind_test_e]
    X, Y = get_training_set(c, f, el, ntr, method,
                            cutoff, nbins, traj, cna_cut)
    return X, Y, X_e, Y_e, x, y, x_e, y_e


def sample_twosets(c1, f1, gc1, en1, el1, c2, f2, gc2, en2, el2, method, ntr, ntest, cutoff, nbins=None, f_e_ratio=100, traj=None, cna_cut=None):
    """ Get training and test set from two databases with method of choice
    """
    ind_test = np.random.choice(np.arange(len(c2)), size=ntest, replace=False)
    ind_test_e = np.random.choice(
        np.arange(len(gc2)), size=ntest//f_e_ratio+1, replace=False)
    ind_train_e = np.random.choice(
        np.arange(len(gc1)), size=ntr//f_e_ratio + 1, replace=False)

    if ((en1 == None).any()) or len(c1) != len(en1):
        en1 = np.zeros(len(c1))
    if ((en2 == None).any()) or len(c2) != len(en2):
        en2 = np.zeros(len(c2))
    X_e, Y_e = gc1[ind_train_e], en1[ind_train_e]
    x, y, x_e, y_e = c2[ind_test], f2[ind_test], gc2[ind_test_e], en2[ind_test_e]
    X, Y = get_training_set(c1, f1, el1, ntr, method,
                            cutoff, nbins, traj, cna_cut)

    return X, Y, X_e, Y_e, x, y, x_e, y_e


def get_right_grid(c, el, nbins, cutoff, method):
    """ Function used to determine the right sampling method to call
    """
    if method == "grid2":
        if len(set(el)) == 1:
            ind = grid_2b_onesp(c, nbins, cutoff)
        elif len(set(el)) >= 2:
            ind = grid_2b_manysp(c, nbins, cutoff, el)
        else:
            print("Error: Number of elements less than 1.")
            return 0

    elif method == "grid3":
        if len(set(el)) == 1:
            ind = grid_3b_onesp(c, nbins, cutoff)
        elif len(set(el)) >= 2:
            ind = grid_3b_manysp(c, nbins, cutoff, el)
        else:
            print("Error: Number of elements less than 1.")
            return 0

    return ind


def get_right_nbins(c, el, cutoff, method, ntr):
    """ Estimate how many bins are necesaary to roughly obtain the desired
        number of training points via linear interpolation at two estimates.
    """
    if method == "grid2":
        ind_1000 = get_right_grid(c, el, 1000, cutoff, method)
        ind_3000 = get_right_grid(c, el, 3000, cutoff, method)
        x = np.array([1000.0, 3000.0])
        y = np.array([len(ind_1000), len(ind_3000)])
        x = x[:, np.newaxis]
        slope, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        nbins = int(ntr/slope)

    elif method == "grid3":
        ind_10 = get_right_grid(c, el, 10, cutoff, method)
        ind_20 = get_right_grid(c, el, 20, cutoff, method)
        x = np.array([10**3, 20**3])
        y = np.array([len(ind_10), len(ind_20)])
        x = x[:, np.newaxis]
        slope, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        nbins = int((ntr/slope)**(1/3.0))

    return nbins


def transform_cna(cna, meaningful_cnas):
    """ Given a cna and a list containing the tuples defining the 
    meaningful cnas, this method returns an array with length equal to the
    total number of cnas which contains the number of times each cna occurrs.
    """
    result = np.zeros(len(meaningful_cnas))
    for i, key in enumerate(meaningful_cnas):
        try:
            result[i] = cna[key]
        except KeyError:
            result[i] = 0
    return result


def get_atomic_cnas(traj, meaningful_cnas, r_cut):
    """ Given a trajectory, a list of tuples containing the CNAS the user is interested
    in, and a cutoff radius, this returns, for every atom for every snapshot in traj, 
    an array which contians the count of cnas for that atom, arranged accordin to the order
    found in meaningful_cnas
    """
    transformed_cna = np.zeros((len(traj)*len(traj[0]), len(meaningful_cnas)))
    for j, atoms in enumerate(traj):
        cna = FullCNA(atoms, r_cut)
        atoms.set_cell([[100, 0, 0], [0, 100, 0], [0, 0, 100]])
        snapshot_cna = cna.get_normal_cna()
        for i, atomic_cna in enumerate(snapshot_cna):
            transformed_cna[j*len(traj[0]) +
                            i] = transform_cna(atomic_cna, meaningful_cnas)
    return transformed_cna


def get_all_cnas(traj, cna_cut):
    """ Given a trajectory and a cutoff radius, returns a dictionary, sorted by value,
    with all the cnas that appear in the trajectyory as keys and the number
    of times they appear as value.
    """
    all_cnas = {}
    for j, atoms in enumerate(traj):
        cna = FullCNA(atoms, cna_cut)
        atoms.set_cell([[100, 0, 0], [0, 100, 0], [0, 0, 100]])
        snapshot_cna = cna.get_normal_cna()
        for i, atomic_cna in enumerate(snapshot_cna):
            for key in atomic_cna:
                try:
                    all_cnas[key] += atomic_cna[key]
                except KeyError:
                    all_cnas[key] = atomic_cna[key]

    sorted_cnas = sorted(all_cnas.items(), key=lambda kv: kv[1])
    sorted_cnas_dict = {}
    for t in sorted_cnas:
        sorted_cnas_dict[t[0]] = t[1]

    return sorted_cnas_dict


def extract_cnas(traj, cna_cut):
    """ Get all the cnas in the trajectory file, then extract the atomic CNA signatures for each CNA present.
    For each atom, the atomic cnas contains a row entry with dimensionality equal to the number of cnas present in the trajectory.
    The all_cnas variable contains a count of occurrance of each cna in the trajectory.
    """
    all_cnas = get_all_cnas(traj, cna_cut)
    atomic_cnas = get_atomic_cnas(traj, all_cnas, cna_cut)
    return atomic_cnas, all_cnas


def sample_uniform_cna(ntr, transformed_cnas):
    """ Sample from an array of transformed cnas a ntr number of indexes.
        For each cnas class, ntr//len(cna classes) atoms are selected which do
        contain at least one pair of that particular class.
    """
    tr_ind = []
    sampled_atoms = np.ones(len(transformed_cnas), dtype='bool')
    ntr_sampled = 0
    for i in range(transformed_cnas.shape[1]):
        indx_this_class = np.where(
            transformed_cnas[:, i][sampled_atoms] > 0)[0]
        ntr_this_class = min(len(indx_this_class), ntr //
                             transformed_cnas.shape[1])
        sampled_inds = np.random.choice(
            indx_this_class, ntr_this_class, replace=False)
        sampled_atoms[sampled_inds] = False
        tr_ind.extend(sampled_inds)
        ntr_sampled += len(sampled_inds)

    if ntr_sampled < ntr:
        additional_inds = np.random.choice(np.arange(len(transformed_cnas))[
                                           sampled_atoms], ntr-ntr_sampled, replace=False)
        tr_ind.extend(additional_inds)
    return np.array(tr_ind)


def sample_cna(traj, cna_cut, ntr):
    """ From a trajcetory file, calculate CNAS using cna_cut as cutoff, 
        order the classes and sample according to the sample_using_cna method
    """
    traj = read(traj, index=':')
    transformed_cnas, all_cnas = extract_cnas(traj, cna_cut)
    print("CNA classes are: \n", all_cnas)
    training_indexes = sample_uniform_cna(ntr, transformed_cnas)
    return training_indexes


def get_training_set(c, f, el, ntr, method, cutoff, nbins=None, traj=None, cna_cut=None):
    """ Call training set sampling
    """
    if method == "random":
        ind = np.arange(len(c))
        ind_tr = np.random.choice(ind, size=ntr, replace=False)
        X, Y = c[ind_tr], f[ind_tr]

    elif method == "grid2" or method == "grid3":
        from itertools import combinations_with_replacement
        from scipy.spatial.distance import cdist

        if nbins is None:
            nbins = get_right_nbins(c, el, cutoff, method, ntr)
        ind = get_right_grid(c, el, nbins, cutoff, method)
        X, Y = c[ind], f[ind]

    elif method == "cna":
        ind = sample_cna(traj, cna_cut, ntr)
        X, Y, = c[ind], f[ind]

    else:
        print("Training method not understood, using random.")
        ind = np.arange(len(c))
        ind_tr = np.random.choice(ind, size=ntr, replace=False)
        X, Y = c[ind_tr], f[ind_tr]

    return X, Y


def get_model(elements, r_cut, ker, sigma=0.5, theta=0.5, noise=0.001, rep_sig=1, alpha =1, r0=10, sigma_eam = 1):
    """ Load the correct model based on the specifications of kernel and number of elements

    """
    if len(elements) == 1:
        if ker == '2b':
            m = models.TwoBodySingleSpeciesModel(
                element=elements, r_cut=r_cut, sigma=sigma, noise=noise, theta=theta, rep_sig=rep_sig)
        elif ker == '3b':
            m = models.ThreeBodySingleSpeciesModel(
                element=elements, r_cut=r_cut, sigma=sigma, noise=noise, theta=theta)
        elif ker == 'combined':
            m = models.CombinedSingleSpeciesModel(element=elements, r_cut=r_cut, sigma_2b=sigma, sigma_3b=sigma*2,
                                                  noise=noise, theta_2b=theta, theta_3b=theta,  rep_sig=rep_sig)
        elif ker == 'mb':
            m = models.ManyBodySingleSpeciesModel(
                element=elements, r_cut=r_cut, sigma=sigma, noise=noise, theta=theta)
        elif ker == 'eam':
            m = models.EamySingleSpeciesModel(
                element=elements, r_cut=r_cut, sigma=sigma, noise=noise, alpha=alpha, r0=r0)
        elif ker == '23eam':
            m = models.TwoThreeEamSingleSpeciesModel(element=elements, r_cut=r_cut, sigma_2b=sigma, sigma_3b=sigma*2, sigma_eam = sigma_eam,
                                                  noise=noise, theta_2b=theta, theta_3b=theta,  alpha=alpha, r0=r0, rep_sig=rep_sig)
        else:
            print(
                "Kernel Type not understood, available options are 2b, 3b, mb, eam, 23eam or combined.")

    elif len(elements) > 1:
        if ker == '2b':
            m = models.TwoBodyManySpeciesModel(
                elements=elements, r_cut=r_cut, sigma=sigma, noise=noise, theta=theta,  rep_sig=rep_sig)
        elif ker == '3b':
            m = models.ThreeBodyManySpeciesModel(
                elements=elements, r_cut=r_cut, sigma=sigma, noise=noise, theta=theta)
        elif ker == 'combined':
            m = models.CombinedManySpeciesModel(elements=elements, r_cut=r_cut, sigma_2b=sigma, sigma_3b=sigma*2,
                                                noise=noise, theta_2b=theta, theta_3b=theta,  rep_sig=rep_sig)
        elif ker == 'mb':
            m = models.ManyBodyManySpeciesModel(
                elements=elements, r_cut=r_cut, sigma=sigma, noise=noise, theta=theta)
        elif ker == 'eam':
            m = models.EamyManySpeciesModel(
                elements=elements, r_cut=r_cut, sigma=sigma, noise=noise, alpha=alpha, r0=r0)
        elif ker == '23eam':
            m = models.TwoThreeEamManySpeciesModel(elements=elements, r_cut=r_cut, sigma_2b=sigma, sigma_3b=sigma*2, sigma_eam = sigma_eam,
                                                  noise=noise, theta_2b=theta, theta_3b=theta,  alpha=alpha, r0=r0, rep_sig=rep_sig)
        else:
            print(
                "Kernel Type not understood, available options are 2b, 3b, mb, eam, 23eam or combined.")

    else:
        print("Number of elements less than 1, elements must be an array or list with len >=1.")

    return m


def train_right_gp(X, Y, elements_1, kernel, sigma, noise, cutoff, train_folder, X_e=None, Y_e=None, train_mode="force", ncores=1, rep_sig=1):
    """ Train GP module based on train mode, kernel and number of atomic species.
    """

    m = get_model(elements_1, cutoff, kernel,
                  sigma, cutoff/5.0, noise, rep_sig)

    print("Training using %i points on %i cores" % (len(X), ncores))
    tic = time.time()
    if train_mode == "force":
        m.fit(X, Y, ncores=ncores)
    else:
        if not energydefault:
            if train_mode == "energy":
                m.fit_energy(X_e, Y_e, ncores=ncores)
            elif train_mode == "force_and_energy":
                m.fit_force_and_energy(X, Y, X_e, Y_e, ncores=ncores)
            else:
                print("Training mode not understood. Defaulting to force training.")
                m.fit(X, Y, ncores=ncores)
        else:
            print("No energies available. Defaulting to force training.")
            m.fit(X, Y, ncores=ncores)
    toc = time.time()
    print("Seconds for training: %.2f" % (toc-tic))
    # Save the GP
    save_gp(m, train_folder, kernel, cutoff, sigma, noise, len(X)+len(X_e))

    return m


def get_gp(train_folder, X, Y, elements_1, kernel, sigma, noise, cutoff, training_points, X_e=None, Y_e=None, train_mode="force", ncores=1, rep_sig=1):
    """ Try to load the specified model if it exists, otherwise train it.

    """
    try:
        if not isinstance(train_folder, Path):
            train_folder = Path(train_folder)

        gp_name = get_model_name(elements_1, kernel, training_points)
        full_path = train_folder / "models" / gp_name
        m = load_model(full_path)

    except FileNotFoundError:
        m = train_right_gp(X, Y, elements_1, kernel, sigma, noise, cutoff, train_folder,
                           X_e, Y_e, train_mode=train_mode, ncores=ncores, rep_sig=rep_sig)

    return m


def get_model_name(elements, kernel, ntr):
    """ Set the name of the model file

    """
    if kernel == "2b":
        first_name = "TwoBody"
    elif kernel == "3b":
        first_name = "ThreeBody"
    elif kernel == "combined":
        first_name = "Combined"
    elif kernel == "eam":
        first_name = "Eam"
    elif kernel == "23eam":
        first_name = "TwoThreeEam"
    elif kernel == "mb":
        first_name = "ManyBody"

    if len(elements) == 1:
        second_name = "SingleSpecies"
    if len(elements) >= 2:
        second_name = "ManySpecies"
    name = first_name + second_name
    full_name = "MODEL_ker_" + name + "_ntr" + str(ntr) + ".json"
    return full_name


def test_forces(m, x, y, plot=False, ncores=1):
    """ Test forces and report significant statystics on the errors incurred by the GP.
    """

    print("Testing the force prediction on %i configurations" % (len(x)))
    # Predict forces on test configurations
    y_pred = m.predict(x, ncores=ncores)
    y_err = y_pred - y  # Calculate error

    MAEC = np.mean(abs(y_err))     # Mean average error on force components
    # Mean average error on force vector
    MAEF = np.mean(np.sqrt(np.sum(np.square(y_err), axis=1)))
    # Standard deviation of the average error on force vector
    SMAEF = np.std(np.sqrt(np.sum(np.square(y_err), axis=1)))
    MF = np.mean(np.linalg.norm(y, axis=1))  # Meean force value
    RMSEF = np.sqrt(np.mean((y_err) ** 2))  # Root mean square error on force
    print('')
    print('RMSE: {:.4} eV/A'.format(RMSEF))
    print('MAEC: {:.4} eV/A'.format(MAEC))
    print('MAEF : {:.4f} +- {:.4f} eV/A'.format(MAEF, SMAEF))
    print('Relative MAEF: {:.4f} +- {:.4f}'.format(MAEF / MF, SMAEF / MF))

    if plot:
        density_plot(y, y_pred, 'force')

    return MAEC, MAEF, SMAEF, MF, RMSEF


def test_energies(m, x_e, y_e, plot=False, ncores=1):
    """ Test forces and report significant statystics on the errors incurred by the GP.
    """

    print("Testing the energy prediction on %i configurations" % (len(x_e)))
    # Predict forces on test configurations
    y_pred = m.predict_energy(x_e, ncores=ncores)
    y_pred /= len(x_e[0])
    y_e /= len(x_e[0])
    y_err = y_pred - y_e  # Calculate error
    MAE = np.mean(abs(y_err))  # Mean average error on energy
    # Standard deviation of the average error on energy
    SMAE = np.std(abs(y_err))
    RMSE_e = np.sqrt(np.mean((y_err) ** 2))  # Root mean square error on energy

    print('')
    print('Energy RMSE: {:.4} eV/atom'.format(RMSE_e))
    print('Energy MAE : {:.4f} +- {:.4f} eV/atom'.format(MAE, SMAE))

    if plot:
        density_plot(y_e, y_pred, 'energy')

    return MAE, SMAE, RMSE_e


def save_gp(m, folder, kernel, cutoff, sigma, noise, ntr):
    """ Save the model
    """
    if not isinstance(folder, Path):
        folder = Path(folder)
    if not os.path.exists(folder / "models"):
        os.makedirs(folder / "models")
    m.save(folder / "models")


def save_report(MAEC, MAEF, SMAEF, MF, RMSEF, folder, test_folder, kernel, cutoff, sigma, noise, ntr, sampling, MAE=None, SMAE=None, RMSE_e=None):
    """ Save a .json file containing details about the model and the errors it incurred in
    """
    if not isinstance(folder, Path):
        folder = Path(folder)
    if not isinstance(test_folder, Path):
        test_folder = Path(test_folder)
    if not os.path.exists(folder / "results"):
        os.makedirs(folder / "results")

    if test_folder == None or test_folder == "None":
        end_name = "%s_%.2f_%.2f_%.4f_%i.json" % (
            kernel, cutoff, sigma, noise, ntr)

    else:
        test_final_name = test_folder.stem
        end_name = "on_%s_%s_%.2f_%.2f_%.4f_%i.json" % (test_final_name,
            kernel, cutoff, sigma, noise, ntr)

    filename = folder / "results" / end_name
    errors = {
        "kernel": kernel,
        "sampling": sampling,
        "ntr": ntr,
        "cutoff": cutoff,
        "sigma": sigma,
        "noise": noise,
        "MAE_c": MAEC,
        "MAE_f": MAEF,
        "SMAE_f": SMAEF,
        "M_f": MF,
        "RMSE_f": RMSEF,
        "MAE_e": MAE,
        "SMAE_e": SMAE,
        "RMSE_e": RMSE_e
    }
    with open(filename, 'w') as fp:
        json.dump(errors, fp, indent=4)

    print("Saved report of errors.")


def train_and_test_gp(train_folder, traj_filename, cutoff=5.0, test_folder=None,
                      training_points=100, test_points=100,
                      kernel='2b', sigma=0.5, noise=0.001, sampling="random", nbins=None,
                      ncores=1, train_mode="force", test_mode="force", f_e_ratio=100, plot=True, cna_cut=None):
    """ Wrapper function that handles everything startng from a .xyz file and details on the kernel.
        Extracts data, creates model, trains GP model and then tests it.
    """
    # Get data, and create training and test sets
    elements_1, confs_1, forces_1, energies_1, global_confs_1 = get_data(
        train_folder, cutoff, traj_filename)
    if test_folder is not None and test_folder != "None":
        print("The test folder is", test_folder)
        elements_2, confs_2, forces_2, energies_2, global_confs_2 = get_data(
            test_folder, cutoff, traj_filename)
        X, Y, X_e, Y_e, x, y, x_e, y_e = sample_twosets(confs_1, forces_1, global_confs_1,
                                                        energies_1, elements_1, confs_2, forces_2, global_confs_2, energies_2,
                                                        elements_2, sampling, training_points, test_points, cutoff, nbins, f_e_ratio, train_folder + '/' + traj_filename, cna_cut)
    else:
        X, Y, X_e, Y_e, x, y, x_e, y_e = sample_oneset(confs_1, forces_1, global_confs_1,
                                                       energies_1, elements_1, sampling, training_points, test_points, cutoff, nbins, f_e_ratio, train_folder + '/' + traj_filename, cna_cut)

    # See if the GP is aleady there, if not train the Gaussian Process
    m = get_gp(train_folder, X, Y, elements_1, kernel, sigma, noise,
               cutoff, training_points, X_e, Y_e, train_mode, ncores)

    # Test the GP
    MAE_c, MAE_f, SMAE_f, M_f, RMSE_f, MAE_e, SMAE_e, RMSE_e = test_gp(
        m, x, y, x_e, y_e, plot, test_mode, ncores)

    # Save a report of the errors
    save_report(MAE_c, MAE_f, SMAE_f, M_f, RMSE_f, train_folder, test_folder, kernel,
                cutoff, sigma, noise, len(X), sampling, MAE_e, SMAE_e, RMSE_e)
    return m


def test_gp(m, x=None, y=None, x_e=None, y_e=None, plot=False, test_mode="forces", ncores=1):
    """ Wrapper function that tests a GP on a test set and returns error metrics.

    """
    if test_mode == "force":
        MAE_c, MAE_f, SMAE_f, M_f, RMSE_f = test_forces(m, x, y, plot, ncores)
        MAE_e, SMAE_e, RMSE_e = None, None, None
    elif test_mode == "energy":
        MAE_c, MAE_f, SMAE_f, M_f, RMSE_f = None, None, None, None
        MAE_e, SMAE_e, RMSE_e = test_energies(m, x_e, y_e, plot, ncores)
    elif test_mode == "force_and_energy":
        MAE_c, MAE_f, SMAE_f, M_f, RMSE_f = test_forces(m, x, y, plot, ncores)
        MAE_e, SMAE_e, RMSE_e = test_energies(m, x_e, y_e, plot, ncores)
    else:
        print("Test mode not understood, use either force, energy or force_and_energy")
    return MAE_c, MAE_f, SMAE_f, M_f, RMSE_f, MAE_e, SMAE_e, RMSE_e


def load_model(filename):
    """ Load GP module based on train mode, kernel and number of atomic species.
    """

    with open(filename) as json_file:
        metadata = json.load(json_file)

    model = metadata['model']
    if model == "TwoBodySingleSpeciesModel":
        m = models.TwoBodySingleSpeciesModel.from_json(filename)
    elif model == "ThreeBodySingleSpeciesModel":
        m = models.ThreeBodySingleSpeciesModel.from_json(filename)
    elif model == "CombinedSingleSpeciesModel":
        m = models.CombinedSingleSpeciesModel.from_json(filename)
    elif model == "EamSingleSpeciesModel":
        m = models.EamSingleSpeciesModel.from_json(filename)
    elif model == "TwoThreeEamSingleSpeciesModel":
        m = models.TwoThreeEamSingleSpeciesModel.from_json(filename)
    elif model == "TwoBodyManySpeciesModel":
        m = models.TwoBodyManySpeciesModel.from_json(filename)
    elif model == "ThreeBodyManySpeciesModel":
        m = models.ThreeBodyManySpeciesModel.from_json(filename)
    elif model == "CombinedManySpeciesModel":
        m = models.CombinedManySpeciesModel.from_json(filename)
    elif model == "EamSingleSpeciesModel":
        m = models.EamSingleSpeciesModel.from_json(filename)
    elif model == "EamManySpeciesModel":
        m = models.EamManySpeciesModel.from_json(filename)
    elif model == "TwoThreeEamSingleSpeciesModel":
        m = models.TwoThreeEamSingleSpeciesModel.from_json(filename)
    elif model == "TwoThreeEamManySpeciesModel":
        m = models.TwoThreeEamManySpeciesModel.from_json(filename)
    else:
        print("Json file does contain unexpected model name")
        return 0
    return m


def density_plot(x, y, mode):
    """ Plot a scatter plot where a gaussian kde has been superimposed in order to
    highlight areas where points are more dense.

    """
    from matplotlib import pyplot as plt
    from scipy.stats import gaussian_kde
    # Calculate the point density
    x = np.ravel(x)
    y = np.ravel(y)
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    plt.scatter(x, y, c=z, s=50, edgecolor='')
    plt.colorbar()
    plt.plot(x, x, 'k-')

    if mode == 'force':
        plt.xlabel(r"True Force [eV/$\AA$]")
        plt.ylabel(r"Predicted Force [eV/$\AA$]")
        plt.title(r"Force Prediction Error")

    elif mode == 'energy':
        plt.xlabel("True Energy [eV/atom]")
        plt.ylabel("Predicted Energy [eV/atom]")
        plt.title("Energy Prediction Error")
    plt.show()


def get_calculator(filepath):

    from mff import calculators
    m = load_model(filepath)
    with open(filepath) as f:
        model_json = json.load(f)
    model_name = model_json['model']

    if model_name == 'TwoBodySingleSpeciesModel':
        calc = calculators.TwoBodySingleSpecies(m.r_cut, m.grid)
    elif model_name == 'ThreeBodySingleSpeciesModel':
        calc = calculators.ThreeBodySingleSpecies(m.r_cut, m.grid)
    elif model_name == 'CombinedSingleSpeciesModel':
        calc = calculators.CombinedSingleSpecies(m.r_cut, m.grid_2b, m.grid_3b)
    elif model_name == 'TwoThreeEamSingleSpeciesModel':
        calc = calculators.TwoThreeEamSingleSpecies(m.r_cut, m.grid_2b, m.grid_3b, m.grid_eam,
            m.gp_eam.kernel.theta[2], m.gp_eam.kernel.theta[3])

    elif model_name == 'TwoBodyManySpeciesModel':
        calc = calculators.TwoBodyManySpecies(m.r_cut,m.elements, m.grid)
    elif model_name == 'ThreeBodyManySpeciesModel':
        calc = calculators.ThreeBodySManySpecies(m.r_cut,m.elements, m.grid)
    elif model_name == 'CombinedManySpeciesModel':
        calc = calculators.CombinedManySpecies(m.r_cut, m.elements, m.grid_2b, m.grid_3b)
    elif model_name == 'TwoThreeEamManySpeciesModel':
        calc = calculators.TwoThreeEamManySpecies(m.r_cut, m.elements, m.grid_2b, m.grid_3b,
             m.grid_eam, m.gp_eam.kernel.theta[2], m.gp_eam.kernel.theta[3])
    else:
        print("ERROR: Model type not understood when loading")
        quit()

    return calc
