import numpy as np
from scipy import interpolate
from itertools import combinations
from mlfmapping import Spline1D, Spline3D

D = 3

class MBExp:

    def __init__(self):
        None
        
    @staticmethod
    def vectorize(rs):
        n, m = rs.shape
        dtype = rs.dtype

        # number of outputs
        n_out = n * (n - 1) / 2

        # Allocate arrays
        r1 = np.zeros([n_out, 1], dtype=dtype)
        r2 = np.zeros([n_out, 1], dtype=dtype)
        r3 = np.zeros([n_out, 1], dtype=dtype)

        r1_hat = np.zeros([n_out, m], dtype=dtype)
        r2_hat = np.zeros([n_out, m], dtype=dtype)

        ds = np.sqrt(np.einsum('nd, nd -> n', rs, rs))
        rs_hat = np.einsum('nd, n -> nd', rs, 1. / ds)

        for i, ((r1_i, r1_hat_i), (r2_i, r2_hat_i)) in enumerate(combinations(zip(ds, rs_hat), r=2)):
            r1[i], r2[i] = r1_i, r2_i
            r1_hat[i, :], r2_hat[i, :] = r1_hat_i, r2_hat_i
            r3[i] = np.linalg.norm(r1_i*r1_hat_i-r2_i*r2_hat_i)

        return r1, r2, r1_hat, r2_hat, r3
	
    def initialize(self,remap_name,):
		remaps = np.load(remap_name)
		self.element1 = remaps[1]
		self.element2 = remaps[2]
		if len(remaps) == 5:
			self.monoelement = True
			self.interp_11 = Spline1D.from_file(remaps[3])
			self.interp_111 = Spline3D.from_file(remaps[4])
			self.element = 
		elif len(remaps) == 10:
			self.monoelement = False
			self.interp_11 = Spline1D.from_file(remaps[3])
			self.interp_12 = Spline1D.from_file(remaps[4])
			self.interp_22 = Spline1D.from_file(remaps[5])
			self.interp_111 = Spline3D.from_file(remaps[6])
			self.interp_112 = Spline3D.from_file(remaps[7])
			self.interp_122 = Spline3D.from_file(remaps[8])
			self.interp_222 = Spline3D.from_file(remaps[9])
		else:
			print("Number of mapped force fields does not match mono or bi-elemental systems, please check %s" %(remap_name))
			quit()
			
		
    def tri_E_forces_confs(self, confs):                 # Forces as a function of configurations
        forces = np.zeros((len(confs), D))
        if self.element2 == None:
			for c in np.arange(len(confs)):
			   rs = np.array((confs[c]))	   
			   force_3 = self.tri_interp.ev_forces(rs) 
			   force_2 = self.pair_interp.ev_forces(rs)
			   forces[c] = force_3 + force_2
        else:
			print("Multi element interpolation not supported yet")
        
        return forces
        
    def tri_E_energies_confs(self, confs):                 # Energies as a function of configurations
        energies = np.zeros(len(confs))
        if self.element2 == None:
			for c in np.arange(len(confs)):
			   rs = np.array((confs[c]))	   
			   energy_3 = self.tri_interp.ev_energy(rs) 
			   energy_2 = self.pair_interp.ev_energy(rs)
			   energies[c] = energy_2/2.0 + energy_3/3.0
        else:
			print("Multi element interpolation not supported yet")

        return energies
     

