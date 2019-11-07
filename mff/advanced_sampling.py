from mff.gp import GaussianProcess
from itertools import product, combinations_with_replacement
from scipy.spatial.distance import cdist
from mff import kernels
from mff.configurations import carve_from_snapshot
from mff.models import TwoBodyTwoSpeciesModel,  CombinedTwoSpeciesModel
from mff.models import TwoBodySingleSpeciesModel,  CombinedSingleSpeciesModel
from pathlib import Path
import logging
import numpy as np
import time
import sys
import random
sys.path.insert(0, '../')


try:
    from skbayes.rvm_ard_models import RVR
    from sklearn.metrics import mean_squared_error
except:
    print("No skbayes module found, rvm sampling cannot be used")

logger = logging.getLogger(__name__)


class Sampling(object):
    """ Sampling methods class
    Class containing sampling methods to optimize the trainng database selection.
    The class is currently set in order to work with local atomic energies, 
    and is therefore made to be used in confined systems (nanoclusters, molecules).
    Some of the mothods used can be applied to force training too (ivm, random), 
    or are independent to the training outputs (grid).
    These methods can be used on systems with PBCs where a local energy is not well defined.
    The class also initializes two GP objects to use in some of its methods.

    Args:
        confs (list of arrays): List of the configurations as M*5 arrays
        energies (array): Local atomic energies, one per configuration
        forces (array): Forces acting on the central atoms of confs, one per configuration
        sigma_2b (float): Lengthscale parameter of the 2-body kernels in Amstrongs
        sigma_3b (float): Lengthscale parameter of the 3-body kernels in Amstrongs
        sigma_mb (float): Lengthscale parameter of the many-body kernel in Amstrongs
        noise (float): Regularization parameter of the Gaussian process
        r_cut (float): Cutoff function for the Gaussian process
        theta (float): Decay lengthscale of the cutoff function for the Gaussian process

    Attributes:
        elements (list): List of the atomic number of the atoms present in the system
        natoms (int): Number of atoms in the system, used for nanoclusters
        K2 (array): Gram matrix for the energy-energy 2-body kernel using the full reduced dataset
        K3 (array): Gram matrix for the energy-energy 3-body kernel using the full reduced dataset

    """

    def __init__(self, confs=None, energies=None,
                 forces=None, sigma_2b=0.05, sigma_3b=0.1, sigma_mb=0.2, noise=0.001, r_cut=8.5, theta=0.5):

        self.confs = confs
        self.energies = energies
        self.forces = forces
        natoms = len(confs[0]) + 1
        self.elements = list(
            np.sort(list(set(confs[0][:, 3:].flatten().tolist()))))
        self.natoms = natoms
        self.K2 = None
        self.K3 = None
        self.sigma_2b, self.sigma_3b, self.sigma_mb, self.noise, self.r_cut, self.theta = (
            sigma_2b, sigma_3b, sigma_mb, noise, r_cut, theta)
        self.get_the_right_kernel('2b')
        self.get_the_right_kernel('3b')


#     def read_xyz(self, filename, r_cut, randomized = True, shuffling = True, forces_label=None, energy_label=None):
#         from ase.io import read
#         traj = read(filename, index=slice(None), format='extxyz')
#         confs, forces, energies = [], [], []
#         for i in np.arange(len(traj)):
#             if randomized:
#                 rand = np.random.randint(0, len(traj[i]), 1)
#             else:
#                 rand = 0
#             co, fe, en = carve_from_snapshot(traj[i], rand, r_cut, forces_label=forces_label, energy_label=energy_label)
#             if len(co[0]) == self.natoms - 1:
#                 confs.append(co[0])
#                 forces.append(fe)
#                 energies.append(en)

#         confs = np.reshape(confs, (len(confs), self.natoms-1, 5))
#         forces = np.reshape(forces, (len(forces), 3))

#         # Bring energies to zero mean
#         energies = np.reshape(energies, len(energies))
#         energies -= np.mean(energies)

#         if shuffling:
#             shuffled_order = np.arange(len(energies))
#             random.shuffle(shuffled_order)
#             energies, forces, confs = energies[shuffled_order], forces[shuffled_order], confs[shuffled_order]

#         self.reduced_energies = energies
#         self.reduced_forces = forces
#         self.reduced_confs = confs
#         del confs, energies, forces, shuffled_order, traj

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def clean_dataset(self, randomized=True, shuffling=True):
        ''' 
        Function used to subsample from a complete trajectory only one atomic environment
        per snapshot. This is necessary when training on energies of nanoclusters in order to assign
        an unique energy value to every configuration and to avoid using redundant information 
        in the form of local atomic environments centered around different atoms in the same snapshot.

        Args:
            randomized (bool): If True, an atom at random is chosen every snapshot, if false always the first
                atom in the configurations will be chosen to represent said snapshot.
            shuffling (bool): if True, once the dataset is created, it is shuffled randomly in order to 
                avoid any bias during incremental training set optimization methods (e.g. rvm, cur, ivm).

        '''
        confs, energies, forces = self.confs, self.energies, self.forces
        natoms = self.natoms

        # Bring energies to zero mean
        energies = np.reshape(energies, len(energies))
        energies -= np.mean(energies)

        # Transform confs into a numpy array
        arrayed_confs = np.zeros((len(forces), natoms-1, 5))
        for i in np.arange(len(confs)):
            try:
                arrayed_confs[i] = confs[i][:natoms-1, :]
            except:
                print("Number of atoms in the configurations is not the expected one")
                arrayed_confs[i] = np.zeros((natoms-1, 5))
                energies[i] = 0
                forces[i] = np.zeros(3)

        # Extract one conf, energy and force per snapshot
        # The particular atom can be chosen at random (random = True)
        # or be always the same (random = False).
        reduced_energies = np.zeros(len(confs)//natoms)
        reduced_confs = np.zeros((len(confs)//natoms, natoms-1, 5))
        reduced_forces = np.zeros((len(confs)//natoms, 3))
        for i in np.arange(len(confs)//natoms):
            if randomized:
                rand = np.random.randint(0, natoms, 1)
            else:
                rand = 0
            reduced_confs[i] = arrayed_confs[i*natoms+rand]
            reduced_energies[i] = energies[i*natoms+rand]
            reduced_forces[i] = forces[i*natoms+rand]

        if shuffling:
            shuffled_order = np.arange(len(reduced_energies))
            random.shuffle(shuffled_order)
            reduced_energies, reduced_forces, reduced_confs = (
                reduced_energies[shuffled_order], reduced_forces[shuffled_order], reduced_confs[shuffled_order])

        # Strip the data of every possible configuration which was discarded because it had the wrong number of atoms
        reduced_energies = reduced_energies[np.where(reduced_energies != 0)]
        reduced_forces = reduced_forces[np.where(reduced_energies != 0)]
        reduced_confs = reduced_confs[np.where(reduced_energies != 0)]

        self.reduced_energies = reduced_energies
        self.reduced_forces = reduced_forces
        self.reduced_confs = reduced_confs
        del confs, energies, forces, natoms, reduced_confs, reduced_forces, reduced_energies, shuffled_order

    def get_the_right_model(self, ker):
        if len(self.elements) == 1:
            if ker == '2b':
                return TwoBodySingleSpeciesModel(self.elements, self.r_cut, self.sigma_2b, self.theta, self.noise)
            elif ker == '3b':
                return CombinedSingleSpeciesModel(element=self.elements, noise=self.noise, sigma_2b=self.sigma_2b, sigma_3b=self.sigma_3b, theta_3b=self.theta, r_cut=self.r_cut, theta_2b=self.theta)
            else:
                print('Kernel type not understood, shutting down')
                return 0

        else:
            if ker == '2b':
                return TwoBodyTwoSpeciesModel(self.elements, self.r_cut, self.sigma_2b, self.theta, self.noise)
            elif ker == '3b':
                return CombinedTwoSpeciesModel(elements=self.elements, noise=self.noise, sigma_2b=self.sigma_2b, sigma_3b=self.sigma_3b, theta_3b=self.theta, r_cut=self.r_cut, theta_2b=self.theta)
            else:
                print('Kernel type not understood, shutting down')
                return 0

    def get_the_right_kernel(self, ker):
        if len(self.elements) == 1:
            if ker == '2b':
                self.gp2 = GaussianProcess(kernel=kernels.TwoBodySingleSpeciesKernel(
                    theta=[self.sigma_2b, self.theta, self.r_cut]), noise=self.noise)
                self.gp2.ncores = 1
            elif ker == '3b':
                self.gp3 = GaussianProcess(kernel=kernels.ThreeBodySingleSpeciesKernel(
                    theta=[self.sigma_3b, self.theta, self.r_cut]), noise=self.noise)
                self.gp3.ncores = 1
            else:
                print('Kernel type not understood, shutting down')
                return 0

        else:
            if ker == '2b':
                self.gp2 = GaussianProcess(kernel=kernels.TwoBodyTwoSpeciesKernel(
                    theta=[self.sigma_2b, self.theta, self.r_cut]), noise=self.noise)
                self.gp2.ncores = 1
            elif ker == '3b':
                self.gp3 = GaussianProcess(kernel=kernels.ThreeBodyTwoSpeciesKernel(
                    theta=[self.sigma_3b, self.theta, self.r_cut]), noise=self.noise)
                self.gp3.ncores = 1
            else:
                print('Kernel type not understood, shutting down')
                return 0

    def train_test_split(self, confs, forces=None, energies=None, ntest=10):
        ''' 
        Function used to subsample a training and a test set: the test set is extracted at random
        and the remaining dataset is trated as a training set (from which we then subsample using the various methods).

        Args:
            confs (array or list): List of the configurations as M*5 arrays
            energies (array): Local atomic energies, one per configuration
            forces (array): Forces acting on the central atoms of confs, one per configuration
            ntest (int): Number of test points, if None, every point that is not a training point will be used
                as a test point

        '''

        if forces is None:
            forces = np.zeros((len(confs), 3))
            print('No forces in the input')

        if energies is None:
            energies = np.zeros(len(confs))
            print('No energies in the input')

        ind = np.arange(len(confs))
        ind_test = np.random.choice(ind, size=ntest, replace=False)
        ind_train = np.array(list(set(ind) - set(ind_test)))
        self.X, self.Y, self.Y_force = confs[ind_train], energies[ind_train], forces[ind_train]
        self.x, self.y, self.y_force = confs[ind_test], energies[ind_test], forces[ind_test]
        del ind, ind_test, ind_train, confs, energies, forces
        try:
            del self.reduced_energies, self.reduced_confs, self.reduced_forces
        except:
            pass

    def ker_2b(self, X1, X2):
        X1, X2 = np.reshape(X1, (self.natoms - 1, 5)
                            ), np.reshape(X2, (self.natoms - 1, 5))
        ker = self.gp2.kernel.k2_ee(
            X1, X2, sig=self.sigma_2b, rc=self.r_cut, theta=self.theta)
        del X1, X2
        return ker

    def ker_3b(self, X1, X2):
        X1, X2 = np.reshape(X1, (self.natoms - 1, 5)
                            ), np.reshape(X2, (self.natoms - 1, 5))
        ker = self.gp3.kernel.k3_ee(
            X1, X2, sig=self.sigma_3b, rc=self.r_cut, theta=self.theta)
        del X1, X2
        return ker

    def normalized_3b(self, X1, X2):
        X1, X2 = np.reshape(X1, (self.natoms - 1, 5)
                            ), np.reshape(X2, (self.natoms - 1, 5))
        ker = self.gp3.kernel.k3_ee(
            X1, X2, sig=self.sigma_3b, rc=self.r_cut, theta=self.theta)
        ker_11 = self.gp3.kernel.k3_ee(
            X1, X1, sig=self.sigma_3b, rc=self.r_cut, theta=self.theta)
        ker_22 = self.gp3.kernel.k3_ee(
            X2, X2, sig=self.sigma_3b, rc=self.r_cut, theta=self.theta)
        ker2 = np.square(ker/np.sqrt(ker_11*ker_22))
        del ker_11, ker_22, X1, X2, ker
        return ker2

    def ker_mb(self, X1, X2):
        X1, X2 = np.reshape(X1, (self.natoms - 1, 5)
                            ), np.reshape(X2, (self.natoms - 1, 5))
        X1, X2 = X1[:, :3], X2[:, :3]
        outer = X1[:, None, :] - X2[None, :, :]
        ker = np.exp(-(np.sum(np.square(outer)/(2.0*self.sigma_mb**2), axis=2)))
        ker = np.einsum('ij -> ', ker)
        del outer, X1, X2
        return ker

    def rvm(self, method='2b',  batchsize=1000):
        ''' 
        Relevance vector machine sampling. This method trains a 2-, 3- or many-body kernel on the energies of the
        partitioned training dataset. The algortihm starts from a dataset containing a batchsize number of training
        configurations extracted from the whole dataset at random. Subsequently, a rvm method is called and a
        variable number of configurations is selected. These are then included in the next batch, and the operation
        is repeated until every point in the training dataset was included at least once.
        The function then returns the indexes of the points returned by the last call of the rvm method.

        Args:
            method (str): 2b or 3b, speciefies which energy kernel to use to calculate the gram matrix
            batchsize (int): number of training points to include in each iteration of the gram matrix calculation

        Returns:
            MAE (float): Mean absolute error made by the final iteration of the method on the test set
            SMAE (float):Standard deviation of the absolute error made by the final iteration of the method on the test set
            RMSE (float): Root mean squared error made by the final iteration of the method on the test set
            index (list): List containing the indexes of all the selected training points
            total_time (float): Excecution time in seconds

        '''
        t0 = time.time()
        if method == '2b':
            rvm = RVR(kernel=self.ker_2b)
        if method == '3b':
            rvm = RVR(kernel=self.ker_3b)
        if method == 'mb':
            rvm = RVR(kernel=self.ker_mb)
        if method == 'normalized_3b':
            rvm = RVR(kernel=self.normalized_3b)

        split = len(self.X)//batchsize + 1    # Decide the number of batches
        # Create a number of evenly sized batches
        batches = np.array_split(range(len(self.X)), split)
        reshaped_X, reshaped_x = np.reshape(self.X, (len(
            self.X), 5*(self.natoms-1))), np.reshape(self.x, (len(self.x), 5*(self.natoms-1)))
        index = []
        for s in np.arange(len(batches)):
            batch_index = list(set(index).union(batches[s]))
            rvm.fit(reshaped_X[batch_index], self.Y[batch_index])
            index = np.asarray(batch_index)[rvm.active_]
        y_hat, var = rvm.predict_dist(reshaped_x)
        error = y_hat - self.y
        MAE = np.mean(np.abs(error))
        SMAE = np.std(np.abs(error))
        RMSE = np.sqrt(np.mean((error) ** 2))
        del var, rvm, split, batches, batch_index, reshaped_X, reshaped_x, y_hat, error
        tf = time.time()
        total_time = tf-t0
        index = list(index)
        return MAE, SMAE, RMSE, index, total_time

    def ivm_e(self, method='2b', ntrain=500, batchsize=1000,  use_pred_error=True, error_metric='energy'):
        '''
        Importance vector machine sampling for energies. This method uses a 2- or 2-body energy kernel and trains 
        it on the energies of the partitioned training dataset. The algortihm starts from two configurations chosen at 
        random. At each iteration, the predicted variance or on the observed error calculated on batchsize configurations
        from the training set is calculated, and the configuration with the highest value is included in the final set.
        The method finishes when ntrain configurations are included in the final set.

        Args:
            method (str): 2b or 3b, speciefies which energy kernel to use to calculate the gram matrix
            ntrain (int): Number of training points to extract from the training dataset
            batchsize (int): number of training points to use in each iteration of the error prediction
            use_pred_error (bool): if true, the predicted variance is used as a metric of the ivm, if false the
                observed error is used instead
            errror_metric (str): specifies whether the final error is calculated on energies or on forces

        Returns:
            MAE (float): Mean absolute error made by the final iteration of the method on the test set
            SMAE (float):Standard deviation of the absolute error made by the final iteration of the method on the test set
            RMSE (float): Root mean squared error made by the final iteration of the method on the test set
            index (list): List containing the indexes of all the selected training points
            total_time (float): Excecution time in seconds

        '''
        t0 = time.time()
        m = self.get_the_right_model(method)
        ndata = len(self.Y)
        mask = np.ones(ndata).astype(bool)
        randints = random.sample(range(ndata), 2)
        m.fit_energy(self.X[randints], self.Y[randints])
        mask[randints] = False
        for i in np.arange(min(ntrain-2, ndata-2)):
            if batchsize > ndata-i-2:
                batchsize = ndata-i-2
            rand_test = random.sample(range(ndata-2-i), batchsize)
            if use_pred_error:
                pred, pred_var = m.predict_energy(
                    self.X[mask][rand_test], return_std=True)
                worst_thing = np.argmax(pred_var)  # L1 norm
            else:
                pred = m.predict_energy(self.X[mask][rand_test])
                worst_thing = np.argmax(
                    abs(pred - self.Y[mask][rand_test]))  # L1 norm
            m.update_energy(self.X[mask][worst_thing],
                            self.Y[mask][worst_thing])
            mask[rand_test[worst_thing]] = False

        if error_metric == 'force':
            y_hat = m.predict(self.x)
            error = y_hat - self.y_force
            MAE = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
            SMAE = np.std(np.sqrt(np.sum(np.square(error), axis=1)))
            RMSE = np.sqrt(np.mean((error) ** 2))
        else:
            y_hat = m.predict_energy(self.x)
            error = y_hat - self.y
            MAE = np.mean(np.abs(error))
            SMAE = np.std(np.abs(error))
            RMSE = np.sqrt(np.mean((error) ** 2))
        index = np.arange(len(self.X))[~mask]
        del mask, worst_thing, pred, rand_test, m, ndata, randints, y_hat
        tf = time.time()
        index = list(index)
        total_time = tf-t0
        return MAE, SMAE, RMSE, index, total_time

    def ivm_f(self, method='2b',  ntrain=500, batchsize=1000, use_pred_error=True, error_metric='energy'):
        '''
        Importance vector machine sampling for forces. This method uses a 2- or 2-body energy kernel and trains 
        it on the energies of the partitioned training dataset. The algortihm starts from two configurations chosen at 
        random. At each iteration, the predicted variance or on the observed error calculated on batchsize configurations
        from the training set is calculated, and the configuration with the highest value is included in the final set.
        The method finishes when ntrain configurations are included in the final set.

        Args:
            method (str): 2b or 3b, speciefies which energy kernel to use to calculate the gram matrix
            ntrain (int): Number of training points to extract from the training dataset
            batchsize (int): number of training points to use in each iteration of the error prediction
            use_pred_error (bool): if true, the predicted variance is used as a metric of the ivm, if false the
                observed error is used instead
            errror_metric (str): specifies whether the final error is calculated on energies or on forces

        Returns:
            MAE (float): Mean absolute error made by the final iteration of the method on the test set
            SMAE (float):Standard deviation of the absolute error made by the final iteration of the method on the test set
            RMSE (float): Root mean squared error made by the final iteration of the method on the test set
            index (list): List containing the indexes of all the selected training points
            total_time (float): Excecution time in seconds

        '''
        t0 = time.time()
        m = self.get_the_right_model(method)
        ndata = len(self.Y_force)
        mask = np.ones(ndata).astype(bool)
        randints = random.sample(range(ndata), 2)
        m.fit(self.X[randints], self.Y_force[randints])
        mask[randints] = False
        for i in np.arange(min(ntrain-2, ndata-2)):
            if batchsize > ndata-i-2:
                batchsize = ndata-i-2
            rand_test = random.sample(range(ndata-2-i), batchsize)
            if use_pred_error:
                pred, pred_var = m.predict(
                    self.X[mask][rand_test], return_std=True)
                worst_thing = np.argmax(np.sum(np.abs(pred_var), axis=1))
                # L1 norm
            else:
                pred = m.predict(self.X[mask][rand_test])
                worst_thing = np.argmax(
                    np.sum(abs(pred - self.Y_force[mask][rand_test]), axis=1))  # L1 norm
            m.update_force(self.X[mask][worst_thing],
                           self.Y_force[mask][worst_thing])
            mask[rand_test[worst_thing]] = False

        if error_metric == 'force':
            y_hat = m.predict(self.x)
            error = y_hat - self.y_force
            MAE = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
            SMAE = np.std(np.sqrt(np.sum(np.square(error), axis=1)))
            RMSE = np.sqrt(np.mean((error) ** 2))
        else:
            y_hat = m.predict_energy(self.x)
            error = y_hat - self.y
            MAE = np.mean(np.abs(error))
            SMAE = np.std(np.abs(error))
            RMSE = np.sqrt(np.mean((error) ** 2))

        index = list(np.arange(len(self.X))[~mask])
        del mask, worst_thing, pred, rand_test, m, ndata, randints, error
        tf = time.time()
        total_time = tf-t0
        return MAE, SMAE, RMSE, index, total_time

    def grid(self, method='2b', nbins=100, error_metric='energy', return_error=True):
        '''
        Grid sampling, based either on interatomic distances (2b) or on triplets of interatomic distances (3b).
        Training configurations are shuffled and are then included in the final database only if they contain a
        distance value (or a triplet of distance values) which is not yet present in the binned histogram of
        distance values (or triplets of distance values) of the final database. This method is very fast since it 
        does not evaluate kernel functions nor gram matrices.

        Args:
            method (str): 2b or 3b, speciefies which energy kernel to use to calculate the gram matrix
            nbins (int): Number of bins to use when building an histogram of interatomic distances.
                If method is 2b, this will specify the value only for distances from the central atom, if
                method is 3b, this will specify the value for triplets of distances.
            errror_metric (str): specifies whether the final error is calculated on energies or on forces
            return_error (bool): if true, error on test set using sampled database is returned

        Returns:
            MAE (float): Mean absolute error made by the final iteration of the method on the test set
            SMAE (float):Standard deviation of the absolute error made by the final iteration of the method on the test set
            RMSE (float): Root mean squared error made by the final iteration of the method on the test set
            index (list): List containing the indexes of all the selected training points
            total_time (float): Excecution time in seconds

        '''
        t0 = time.time()
        if method == '2b':
            if len(self.elements) == 1:
                stored_histogram = np.zeros(nbins)
                index = []
                ind = np.arange(len(self.X))
                randomarange = np.random.choice(
                    ind, size=len(self.X), replace=False)
                for j in randomarange:  # for every snapshot of the trajectory file
                    distances = np.sqrt(
                        np.einsum('id -> i', np.square(self.X[j][:, :3])))
                    distances[np.where(distances > self.r_cut)] = None
                    this_snapshot_histogram = np.histogram(
                        distances, nbins, (0.0, self.r_cut))
                    if (stored_histogram - this_snapshot_histogram[0] < 0).any():
                        index.append(j)
                        stored_histogram += this_snapshot_histogram[0]

                m = TwoBodySingleSpeciesModel(
                    self.elements, self.r_cut, self.sigma_2b, self.theta, self.noise)

            if len(self.elements) == 2:
                stored_histogram = np.zeros((nbins, 3))
                index = []
                ind = np.arange(len(self.X))
                randomarange = np.random.choice(
                    ind, size=len(self.X), replace=False)
                for j in randomarange:  # for every snapshot of the trajectory file
                    distances = np.sqrt(
                        np.einsum('id -> i', np.square(self.X[j][:, :3])))
                    distances[np.where(distances > self.r_cut)] = None
                    element_pairs = list(
                        combinations_with_replacement(self.elements, 2))
                    for k in range(3):
                        if k == 1:  # In the case of two different elements, we have to account for permutation invariance
                            this_element_pair = np.union1d(
                                np.intersect1d(
                                    np.where(self.X[j][:, 3] == element_pairs[k][0]), np.where(self.X[j][:, 4] == element_pairs[k][1])),
                                np.intersect1d(
                                    np.where(self.X[j][:, 3] == element_pairs[k][1]), np.where(self.X[j][:, 4] == element_pairs[k][0])))
                        else:
                            this_element_pair = np.intersect1d(
                                np.where(self.X[j][:, 3] == element_pairs[k][0]), np.where(self.X[j][:, 4] == element_pairs[k][1]))
                        distances_this = distances[this_element_pair]

                        this_snapshot_histogram = np.histogram(
                            distances_this, nbins, (0.0, self.r_cut))
                        if (stored_histogram[:, k] - this_snapshot_histogram[0] < 0).any():
                            index.append(j)
                            stored_histogram[:,
                                             k] += this_snapshot_histogram[0]

                m = TwoBodyTwoSpeciesModel(
                    self.elements, self.r_cut, self.sigma_2b, self.theta, self.noise)

        elif method == '3b':
            if len(self.elements) == 1:
                stored_histogram = np.zeros((nbins, nbins, nbins))
                index = []
                ind = np.arange(len(self.X))
                randomarange = np.random.choice(
                    ind, size=len(self.X), replace=False)
                for j in randomarange:  # for every snapshot of the trajectory file
                    atoms = np.vstack(([0., 0., 0.], self.X[j][:, :3]))
                    distances = cdist(atoms, atoms)
                    distances[np.where(distances > self.r_cut)] = None
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
                                                             range=((0.0, self.r_cut), (0.0, self.r_cut), (0.0, self.r_cut)))

                    if (stored_histogram - this_snapshot_histogram[0] < 0).any():
                        index.append(j)
                        stored_histogram += this_snapshot_histogram[0]

                m = CombinedSingleSpeciesModel(element=self.elements, noise=self.noise, sigma_2b=self.sigma_2b,
                                               sigma_3b=self.sigma_3b, theta_3b=self.theta, r_cut=self.r_cut, theta_2b=self.theta)

            elif len(self.elements) == 2:
                stored_histogram = np.zeros((nbins, nbins, nbins, 4))
                index = []
                ind = np.arange(len(self.X))
                randomarange = np.random.choice(
                    ind, size=len(self.X), replace=False)
                for j in randomarange:  # for every snapshot of the trajectory file
                    atoms = np.vstack(([0., 0., 0.], self.X[j][:, :3]))
                    distances = cdist(atoms, atoms)
                    distances[np.where(distances > self.r_cut)] = None
                    distances[np.where(distances == 0)] = None
                    possible_triplets = list(
                        combinations_with_replacement(self.elements, 3))
                    triplets = []
                    elements_triplets = []
                    for k in np.argwhere(distances[:, 0] > 0):
                        for l in np.argwhere(distances[0, :] > 0):
                            if distances[k, l] > 0:
                                elements_triplets.append(
                                    np.sort([self.X[j][0, 3], self.X[j][k-1, 4], self.X[j][l-1, 4]]))
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
                                                             range=((0.0, self.r_cut), (0.0, self.r_cut), (0.0, self.r_cut)))
                    for k in np.arange(4):
                        valid_triplets = triplets[np.where(
                            elements_triplets == possible_triplets[k]), :][0]
                        this_snapshot_histogram = np.histogramdd(valid_triplets, bins=(nbins, nbins, nbins),
                                                                 range=((0.0, self.r_cut), (0.0, self.r_cut), (0.0, self.r_cut)))

                        if (stored_histogram[:, :, :, k] - this_snapshot_histogram[0] < 0).any():
                            index.append(j)
                            stored_histogram[:, :, :,
                                             k] += this_snapshot_histogram[0]
                m = CombinedTwoSpeciesModel(elements=self.elements, noise=self.noise, sigma_2b=self.sigma_2b,
                                            sigma_3b=self.sigma_3b, theta_3b=self.theta, r_cut=self.r_cut, theta_2b=self.theta)

        else:
            print('Method must be either 2b or 3b')
            return 0

        if return_error:
            if error_metric == 'force':
                m.fit(self.X[index], self.Y_force[index])
                y_hat = m.predict(self.x)
                error = y_hat - self.y_force
                MAE = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
                SMAE = np.std(np.sqrt(np.sum(np.square(error), axis=1)))
                RMSE = np.sqrt(np.mean((error) ** 2))
            else:
                m.fit_energy(self.X[index], self.Y[index])
                y_hat = m.predict_energy(self.x)
                error = y_hat - self.y
                MAE = np.mean(np.abs(error))
                SMAE = np.std(np.abs(error))
                RMSE = np.sqrt(np.mean((error) ** 2))
            del m, distances, this_snapshot_histogram, randomarange, stored_histogram, y_hat, error
            tf = time.time()
            total_time = tf-t0
            index = list(set(index))
            return MAE, SMAE, RMSE, index, total_time

        else:
            index = list(set(index))
            return index

    def cur(self, method='2b', ntrain=1000, batchsize=1000, error_metric='energy'):
        '''
        Sampling using the CUR decomposition technique.
        The complete dataset is first divided into batches, then the energy-energy Gram matrix is
        calculated for each batch. An svd decomposition is subsequently applied to each gram matrix,
        and a number of entries (columns) is selected based on their importance score.
        The method is calibrated so that the final number of training points selected is roughly equal
        to the input parameter ntrain.

        Args:
            method (str): 2b or 3b, speciefies which energy kernel to use to calculate the gram matrix
            ntrain (int): Number of training points to be selected from the whole dataset
            batchsize (int): Number of data points to be used for each calculation of the gram matrix.
                Lower values make the computation faster but the error might be higher.
            errror_metric (str): specifies whether the final error is calculated on energies or on forces

        Returns:
            MAE (float): Mean absolute error made by the final iteration of the method on the test set
            SMAE (float):Standard deviation of the absolute error made by the final iteration of the method on the test set
            RMSE (float): Root mean squared error made by the final iteration of the method on the test set
            index (list): List containing the indexes of all the selected training points
            total_time (float): Excecution time in seconds

        '''

        t0 = time.time()
        split = len(self.X)//batchsize + 1    # Decide the number of batches
        # Create a number of evenly sized batches
        batches = np.array_split(range(len(self.X)), split)
        index = []
        ntr_per_batch = ntrain//split
        for s in np.arange(split):
            #batches[s] = list(batches[s])
            batch_confs = self.X[batches[s]]
            if method == '2b':
                gram = self.gp2.calc_gram_ee(batch_confs)
            elif method == '3b':
                gram = self.gp3.calc_gram_ee(batch_confs)
            else:
                print('Method must be either 2b or 3b')
                return 0

            u, p, v = np.linalg.svd(gram)
            v2 = np.square(v)
            score = np.sum(v2[:ntr_per_batch], axis=0)/ntr_per_batch

            # Calculate the score value of the nth percentile of the score distribution. This is used when randomly selecting the columns
            median = np.percentile(score, int(
                100 - 100*(ntr_per_batch/len(score))))
            std = np.std(score)

            # Choose randomly the columns with probability proportional to a sigmoid function applied to the score, centred on median
            prob = np.minimum(np.ones(len(score)), self.sigmoid(
                (score-median)/std/(ntr_per_batch/len(score))))
            rand = np.random.uniform(size=len(score))
            accepted = np.sign(prob - rand)
            accepted = ((accepted+1)//2).astype(bool)
            watever = batches[s]
            these_ones = watever[accepted]
            index.append(these_ones)

            del score, accepted, prob, median, std, gram, u, s, v

        index = np.concatenate(index).ravel().tolist()

        m = self.get_the_right_model(method)
        if error_metric == 'energy':
            m.fit_energy(self.X[index], self.Y[index])
            y_hat = m.predict_energy(self.x)
            error = y_hat - self.y
            MAE = np.mean(np.abs(error))
            SMAE = np.std(np.abs(error))
            RMSE = np.sqrt(np.mean((error) ** 2))
        else:
            m.fit(self.X[index], self.Y_force[index])
            y_hat = m.predict(self.x)
            error = y_hat - self.y_force
            MAE = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
            SMAE = np.std(np.sqrt(np.sum(np.square(error), axis=1)))
            RMSE = np.sqrt(np.mean((error) ** 2))

        index_return = np.arange(len(self.X))[index]
        del error, y_hat, m
        tf = time.time()
        return MAE, SMAE, RMSE, list(index_return), tf-t0

    def random(self, method='2b', ntrain=500, error_metric='energy', return_error=True):
        '''
        Random subsampling of training points from the larger training dataset.

        Args:
            method (str): 2b or 3b, speciefies which energy kernel to use to calculate the gram matrix
            ntrain (int): Number of points to include in the final dataset.
            errror_metric (str): specifies whether the final error is calculated on energies or on forces
            return_error (bool): if True, train a GP and run a test

        Returns:
            MAE (float): Mean absolute error made by the final iteration of the method on the test set
            SMAE (float):Standard deviation of the absolute error made by the final iteration of the method on the test set
            RMSE (float): Root mean squared error made by the final iteration of the method on the test set
            index (list): List containing the indexes of all the selected training points
            total_time (float): Excecution time in seconds

        '''
        ind = np.arange(len(self.X))
        ind_train = np.random.choice(ind, size=ntrain, replace=False)

        if return_error:
            t0 = time.time()
            train_confs = self.X[ind_train]
            train_energy = self.Y[ind_train]
            train_forces = self.Y_force[ind_train]
            m = self.get_the_right_model(method)
            if error_metric == 'force':
                m.fit(train_confs, train_forces)
                y_hat = m.predict(self.x)
                error = y_hat - self.y_force
                MAE = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
                SMAE = np.std(np.sqrt(np.sum(np.square(error), axis=1)))
                RMSE = np.sqrt(np.mean((error) ** 2))
            else:
                m.fit_energy(train_confs, train_energy)
                y_hat = m.predict_energy(self.x)
                error = y_hat - self.y
                MAE = np.mean(np.abs(error))
                SMAE = np.std(np.abs(error))
                RMSE = np.sqrt(np.mean((error) ** 2))

            del m, train_confs, train_energy, train_forces, error, y_hat

            tf = time.time()
            index = list(ind_train)
            total_time = tf-t0
            return MAE, SMAE, RMSE, index, total_time

        else:
            return list(ind_train)

    def test_forces(self, index, method='2b', sig_2b=0.2, sig_3b=0.8, noise=0.001):
        '''
        Random subsampling of training points from the larger training dataset.

        Args:
            method (str): 2b or 3b, speciefies which energy kernel to use to calculate the gram matrix
            ntrain (int): Number of points to include in the final dataset.
            errror_metric (str): specifies whether the final error is calculated on energies or on forces

        Returns:
            MAE (float): Mean absolute error made by the final iteration of the method on the test set
            SMAE (float):Standard deviation of the absolute error made by the final iteration of the method on the test set
            RMSE (float): Root mean squared error made by the final iteration of the method on the test set
            index (list): List containing the indexes of all the selected training points
            total_time (float): Excecution time in seconds

        '''
        self.sigma_2b, self.sigma_3b, self.noise = sig_2b, sig_3b, noise
        m = self.get_the_right_model(method)
        m.fit(self.X[index], self.Y_force[index])
        y_hat = m.predict(self.x)
        error = self.y_force - y_hat
        MAEF = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
        SMAEF = np.std(np.sqrt(np.sum(np.square(error), axis=1)))
        RMSE = np.sqrt(np.mean((error) ** 2))
        print("MAEF: %.4f SMAEF: %.4f RMSE: %.4f" % (MAEF, SMAEF, RMSE))
        del m, error, y_hat, index
        return MAEF, SMAEF, RMSE
