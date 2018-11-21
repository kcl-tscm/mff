import logging
import numpy as np
import time
from pathlib import Path
from mff.models import TwoBodySingleSpeciesModel,  CombinedSingleSpeciesModel
from mff.models import TwoBodyTwoSpeciesModel,  CombinedTwoSpeciesModel
from mff.gp import GaussianProcess 
from mff import kernels
from skbayes.rvm_ard_models import RVR
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.ERROR)


class Sampling(object):
    """ Sampling methods class
    Class containing sampling methods to optimize the trainng database selection.
    The class is currently set in order to work with local atomic energies, 
    and is therefore made to be used in confined systems (nanoclusters, molecules).
    Some of the mothods used can be applied to force training too (ivm_sampling ), 
    or are independent to the training outputs (grid_2b sampling and grid_3b_sampling).
    These methods can be used on systems with PBCs where a local energy is not well defined.

    Args:
        confs (list of arrays): A kernel object (typically a two or three body)
        noise (float): The regularising noise level (typically named \sigma_n^2)
        optimizer (str): The kind of optimization of marginal likelihood (not implemented yet)

    Attributes:
        X_train_ (list): The configurations used for training
        alpha_ (array): The coefficients obtained during training
        L_ (array): The lower triangular matrix from cholesky decomposition of gram matrix
        K (array): The kernel gram matrix
    """

    def __init__(self, confs=None, energies=None,
                 forces=None):
        
        self.confs = confs
        self.energies = energies
        self.forces = forces
        natoms = len(confs[0]) + 1
        atom_number_list = [confs[0,0,3] for conf in confs]
        self.elements = np.unique(atom_number_list, return_counts=False)

        self.natoms = natoms
        
        
    def clean_dataset(self, random = True):
        confs, energies, forces = self.confs, self.energies, self.forces
        natoms = self.natoms
        
        # Transform confs into a numpy array
        arrayed_confs = np.zeros((len(forces), natoms-1, 5))
        for i in np.arange(len(confs)):
            arrayed_confs[i] = confs[i]
            
        # Bring energies to zero mean
        energies = np.reshape(energies, len(energies))
        energies -= np.mean(energies)

        # Extract one conf, energy and force per snapshot
        # The particular atom can be chosen at random (random = True)
        # or be always the same (random = False).
        reduced_energies = np.zeros(len(confs)//natoms)
        reduced_confs = np.zeros((len(confs)//natoms, natoms-1, 5))
        reduced_forces = np.zeros((len(confs)//natoms, 3))
        for i in np.arange(len(confs)//natoms):
            if random:
                rand = np.random.randint(0, natoms, 1)
            else:
                rand = 0
            reduced_energies[i] = energies[i*natoms+rand]
            reduced_confs[i] = confs[i*natoms+rand]
            reduced_forces[i] = forces[i*natoms+rand]
        
        self.reduced_energies = reduced_energies
        self.reduced_confs = reduced_confs
        self.reduced_forces = reduced_forces
        
        
    def get_the_right_model(self, ker):
        if len(self.elements) == 1:
            if ker == '2b':
                return TwoBodySingleSpeciesModel(self.elements, self.r_cut, self.sigma_2b, self.theta, self.noise)
            elif ker == '3b':
                return CombinedSingleSpeciesModel(element=self.elements, noise=self.noise, sigma_2b=self.sigma_2b
                                                   , sigma_3b=self.sigma_3b, theta_3b=self.theta, r_cut=self.r_cut, theta_2b=self.theta)
            else:
                print('Kernel type not understood, shutting down')
                return 0 
            
        else:
            if ker == '2b':
                return TwoBodyTwoSpeciesModel(self.elements, self.r_cut, self.sigma_2b, self.theta, self.noise)
            elif ker == '3b':
                return CombinedTwoSpeciesModel(elements=self.elements, noise=self.noise, sigma_2b=self.sigma_2b
                                                   , sigma_3b=self.sigma_3b, theta_3b=self.theta, r_cut=self.r_cut, theta_2b=self.theta)
            else:
                print('Kernel type not understood, shutting down')
                return 0      
        
    def get_the_right_kernel(self, ker):
        if len(self.elements) == 1:
            if ker == '2b':
                self.gp2 = GaussianProcess(kernel= kernels.TwoBodySingleSpeciesKernel(theta=[self.sigma_2b, self.theta, self.r_cut]), noise= self.noise)
            elif ker == '3b':
                self.gp3 = GaussianProcess(kernel= kernels.ThreeBodySingleSpeciesKernel(
                    theta=[self.sigma_3b, self.theta, self.r_cut]), noise= self.noise)
            else:
                print('Kernel type not understood, shutting down')
                return 0 
            
        else:
            if ker == '2b':
                self.gp2 = GaussianProcess(kernel= kernels.TwoBodyTwoSpeciesKernel(theta=[self.sigma_2b, self.theta, self.r_cut]), noise= self.noise)
            elif ker == '3b':
                self.gp3 = GaussianProcess(kernel= kernels.ThreeBodyTwoSpeciesKernel(
                    theta=[self.sigma_3b, self.theta, self.r_cut]), noise= noise)
            else:
                print('Kernel type not understood, shutting down')
                return 0     
        
        
    def train_test_split(self, confs=[], forces=[], energies=[], ntr = 0, ntest = None):
        ind = np.arange(len(confs))
        if ntest == None: # Use everything that is not training as test
            ind_tot = np.random.choice(ind, size=len(confs), replace=False)        
        else:
            ind_tot = np.random.choice(ind, size=ntr+ntest, replace=False)
        self.X, self.Y, self.Y_force = confs[ind_tot[:ntr]], energies[ind_tot[:ntr]], forces[ind_tot[:ntr]]
        self.x, self.y, self.y_force = confs[ind_tot[ntr:]], energies[ind_tot[ntr:]], forces[ind_tot[ntr:]]
        
        
    def initialize_gps(self, sigma_2b = 0.05, sigma_3b = 0.1, sigma_mb = 0.2, noise = 0.001, r_cut = 8.5, theta = 0.5):
        self.sigma_2b, self.sigma_3b, self.sigma_mb, self.noise, self.r_cut, self.theta = (
            sigma_2b, sigma_3b, sigma_mb, noise, r_cut, theta)
        self.gp2 = self.get_the_right_kernel('2b')
        self.gp3 = self.get_the_right_kernel('3b')

        
        
    def ker_2b(self, X1, X2):
        X1, X2 = np.reshape(X1, (18,5)), np.reshape(X2, (18,5))
        ker = self.gp2.kernel.k2_ee(X1, X2, sig=self.sigma_2b, rc=self.r_cut, theta=self.theta)
        return ker

    
    def ker_3b(self, X1, X2):
        X1, X2 = np.reshape(X1, (18,5)), np.reshape(X2, (18,5))
        ker = self.gp3.kernel.k3_ee(X1, X2, sig=self.sigma_3b, rc=self.r_cut, theta=self.theta)
        return ker

    
    def soap(self, X1, X2):
        X1, X2 = np.reshape(X1, (18,5)), np.reshape(X2, (18,5))
        ker = self.gp3.kernel.k3_ee(X1, X2, sig=self.sigma_3b, rc=self.r_cut, theta=self.theta)
        ker_11 = self.gp3.kernel.k3_ee(X1, X1, sig=self.sigma_3b, rc=self.r_cut, theta=self.theta)
        ker_22 = self.gp3.kernel.k3_ee(X2, X2, sig=self.sigma_3b, rc=self.r_cut, theta=self.theta)
        return np.square(ker/np.sqrt(ker_11*ker_22))

    
    def ker_mb(self, X1, X2):
        X1, X2 = np.reshape(X1, (18,5)), np.reshape(X2, (18,5))
        X1, X2 = X1[:,:3], X2[:,:3]
        outer = X1[:,None,:] - X2[None, :,:]
        ker = np.exp(-(np.sum(np.square(outer)/(2.0*self.sigma_mb**2), axis = 2)))
        ker = np.einsum('ij -> ', ker)
        return ker       
      
        
    def rvm_sampling(self, ker = '2b'):
        if ker == '2b':
            rvm = RVR(kernel = self.ker_2b)
        if ker == '3b':
            rvm = RVR(kernel = self.ker_3b)
        if ker == 'mb':
            rvm = RVR(kernel = self.ker_mb)
        if ker == 'soap':
            rvm = RVR(kernel = self.soap)
        t1 = time.time()
        reshaped_X, reshaped_x = np.reshape(self.X, (len(self.X), 5*(self.natoms-1))), np.reshape(self.x, (len(self.x), 5*(self.natoms-1)))
        rvm.fit(reshaped_X, self.Y)
        t2 = time.time()
        y_hat,var     = rvm.predict_dist(reshaped_x)
        rvs       = np.sum(rvm.active_)
        print("RVM %s error on test set is %.4f, number of relevant vectors is %i, time %.4f" %(ker, mean_squared_error(y_hat,self.y), rvs, t2 - t1)) 
        return mean_squared_error(y_hat,self.y), y_hat, rvm.active_

    
    def ivm_sampling_energy(self, ker = '2b', threshold_error = 0.002, max_iter = 1000, use_pred_error = True):
        
        m = self.get_the_right_model(ker)
        ndata = len(self.Y)
        mask = np.ones(ndata).astype(bool)
        randints = np.random.randint(0, ndata, 2)
        m.fit_energy(self.X[randints], self.Y[randints])
        mask[randints] = False
        worst_thing = 1.0
        for i in np.arange(min(max_iter, ndata-2)):
            pred = m.predict(self.X[mask])
            pred, pred_var =  m.predict_energy(self.X[mask], return_std = True)
            if use_pred_error:
                worst_thing = np.argmax(pred_var)  # L1 norm
            else:
                worst_thing = np.argmax(abs(pred - self.Y[mask]))  # L1 norm
            m.update_energy(self.X[mask][worst_thing], self.Y[mask][worst_thing])
            mask[mask][worst_thing] = False
            if(max(pred_var) < threshold_error):
                break
        y_hat = m.predict_energy(self.x)
        error = mean_squared_error(y_hat, self.y)
        return error, y_hat, [not i for i in mask]

    def ivm_sampling_force(self, ker = '2b', threshold_error = 0.002, max_iter = 1000, use_pred_error = True):
        m = self.get_the_right_model(ker)
        ndata = len(self.Y_force)
        mask = np.ones(ndata).astype(bool)
        randints = np.random.randint(0, ndata, 2)
        m.fit(self.X[randints], self.Y_force[randints])
        mask[randints] = False
        worst_thing = 1.0
        for i in np.arange(min(max_iter, ndata-2)):
            pred = m.predict(self.X[mask])
            pred, pred_var =  m.predict(self.X[mask], return_std = True)
            if use_pred_error:
                worst_thing = np.argmax(np.sum(np.abs(pred_var), axis = 1))  # L1 norm
            else:
                worst_thing = np.argmax(np.sum(abs(pred - self.Y_force[mask]), axis = 1))  # L1 norm
            m.update(self.X[mask][worst_thing], self.Y_force[mask][worst_thing])
            mask[mask][worst_thing] = False
            if(max(pred_var) < threshold_error):
                break
        y_hat = m.predict(self.x)
        error = mean_squared_error(y_hat, self.Y_force)
        return error, y_hat, [not i for i in mask]
    
    def grid_2b_sampling(self, nbins):
        stored_histogram = np.zeros(nbins)
        index = []
        ind = np.arange(len(self.X))
        randomarange = np.random.choice(ind, size=len(self.X), replace=False)
        for j in randomarange: # for every snapshot of the trajectory file
            distances = np.sqrt(np.einsum('id -> d', np.square(self.X[j,:,:3])))
            distances[np.where(distances > self.r_cut)] = None
            this_snapshot_histogram = np.histogram(distances, nbins, (0.0, self.r_cut))
            if (stored_histogram - this_snapshot_histogram[0] < 0).any():
                index.append(j)
                stored_histogram += this_snapshot_histogram[0]
                
        m = TwoBodySingleSpeciesModel(self.elements, self.r_cut, self.sigma_2b, self.theta, self.noise)
        m.fit_energy(self.X[index], self.Y[index])
        y_hat = m.predict_energy(self.x)
        error = mean_squared_error(y_hat, self.y)
        return error, y_hat, np.sort(index)
    
    
    def grid_3b_sampling(self, nbins):
        stored_histogram = np.zeros((nbins, nbins, nbins))
        index = []
        ind = np.arange(len(self.X))
        randomarange = np.random.choice(ind, size=len(self.X), replace=False)
        for j in randomarange: # for every snapshot of the trajectory file
            atoms = np.vstack(([0., 0., 0.], self.X[j][:,:3]))
            distances  = cdist(atoms, atoms)
            distances[np.where(distances > self.r_cut)] = None
            distances[np.where(distances == 0 )] = None
            triplets = []
            for k in np.argwhere(distances[:,0] > 0 ):
                for l in np.argwhere(distances[0,:] > 0 ):
                    if distances[k,l] > 0 :
                        triplets.append([distances[0, k], distances[0, l], distances[k, l]])
                        triplets.append([distances[0, l], distances[k, l], distances[0, k]])
                        triplets.append([distances[k, l], distances[0, k], distances[0, l]])
                        
            triplets = np.reshape(triplets, (len(triplets), 3)) 
            this_snapshot_histogram = np.histogramdd(triplets, bins = (nbins, nbins, nbins), 
                                                     range =  ((0.0, self.r_cut), (0.0, self.r_cut), (0.0, self.r_cut)))
            
            if (stored_histogram - this_snapshot_histogram[0] < 0).any():
                index.append(j)
                stored_histogram += this_snapshot_histogram[0]
                
        m = CombinedTwoSpeciesModel(elements=self.elements, noise=self.noise, sigma_2b=self.sigma_2b
                                               , sigma_3b=self.sigma_3b, theta_3b=self.theta, r_cut=self.r_cut, theta_2b=self.theta)
        m.fit_energy(self.X[index], self.Y[index])
        y_hat = m.predict_energy(self.x)
        error = mean_squared_error(y_hat, self.y)
        return error, y_hat, np.sort(index)
    
    
    def random_sampling(self, ker = '2b'):
        
        m = self.get_the_right_model(ker)
        m.fit_energy(self.X, self.Y)
        y_hat = m.predict_energy(self.x)
        error = mean_squared_error(y_hat, self.y)
        return error, y_hat, np.arange(len(self.X))


    def test_gp_on_forces(self, index, ker = '2b'):
        
        m = self.get_the_right_model(ker)
        m.fit(self.X[index], self.Y_force[index])
        y_hat = m.predict(self.x)
        error = self.y_force - y_hat
        MAEF = np.mean(np.sqrt(np.sum(np.square(error), axis=1)))
        SMAEF = np.std(np.sqrt(np.sum(np.square(error), axis=1)))
        RMSE = np.sqrt(np.mean((error) ** 2))   
        print("MAEF: %.4f SMAEF: %.4f RMSE: %.4f" %(MAEF, SMAEF, RMSE))
        return MAEF, SMAEF, RMSE
    
    
    