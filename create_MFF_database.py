from ase.io import extxyz 
import numpy as np
from ase.calculators.neighborlist import NeighborList
from ase.geometry import find_mic

def carve_confs(filename, r_cut):

  	### Open file and get number of atoms and steps ###
	f = open(filename, 'r')
	N = int(f.readline())
	
	num_lines = 1+sum(1 for line in f)
	f.close()
	steps = num_lines/(N+2) 
	n_data = float(min(5000, N*steps))
	print('Database will have %i entries' %(n_data))

	### Read the number and types of elements ###
	atoms = extxyz.read_extxyz(filename, index = 0)
	atoms = next(atoms)
	elementslist = list(set(atoms.get_atomic_numbers()))
	elplace = []

	for i in np.arange(len(elementslist)):
		elplace.append(np.where(atoms.get_atomic_numbers() == elementslist[i]))
	elplace = np.array(elplace)
	
	confs = []
	forces = []
	energies = []
	
	if len(elementslist) == 2:
		print("There are 2 elements in the XYZ file")
		### Choose the number of entries each element will have, proportional to the square root of the ratio of occurrences of each element ###
		ratio = np.sqrt(len(elplace[0,0])/float(len(elplace[1,0])))
		nc_el1 = int(n_data*ratio/(1.0+ratio))
		nc_el2 = n_data - nc_el1

		
		cutoffs = np.ones(N)*r_cut/2.         
		nl = NeighborList(cutoffs, skin=0., sorted=False, self_interaction=False,bothways=True)

		### Build conf and forces database centered on element 1 ###
		for i in np.arange(nc_el1):
			print('step %i' %(i))
			j = int(i*float(steps)/nc_el1)
			atoms = extxyz.read_extxyz(filename, index = j)
			atoms = next(atoms)
			nl.build(atoms)
			cell = atoms.get_cell()
			ind_atom = int(i%len(elplace[0,0]))		# Select the atom number rotationg between the total atoms of el 1 present #

			d = np.array([atoms.arrays['positions'][elplace[0,0]][ind_atom]])
			errvalue = 0
			try:
				force = atoms.get_array('force')[elplace[0,0]][ind_atom]   
			except KeyError:
				print('Forces in the xyz file are not present, or are not called force')
				force = None
				errvalue += 1
			try:
				energy = atoms.get_array('energy')[elplace[0,0]][ind_atom]
			except KeyError:
				print('Energies in the xyz file are not present, or are not called energy')
				energy = None
				errvalue += 1
				
			if errvalue ==2:
				print('Cannot find energy or force values in the xyz file, shutting down now')
				quit()
				
			indices, offsets = nl.get_neighbors(ind_atom)
			offsets = np.dot(offsets, cell)
			conf = np.zeros((len(indices), 5))
				
			for k, (a2, offset) in enumerate(zip(indices, offsets)):
				d = atoms.positions[a2] + offset - atoms.positions[elplace[0,0]][ind_atom]
				conf[k,:3] = d
				conf[k,4] = atoms.get_atomic_numbers()[a2]					# Set the last digit of confs to be the element of the atom in the conf

			conf[:,3] = atoms.get_atomic_numbers()[elplace[0,0]][ind_atom]  # Set the fourth digit of confs to be the element of the central atom
			confs.append(conf)
			forces.append(force)
			energies.append(energy)
			
		### Build conf and forces database centered on element 2, exact same procedure ###
		for i in np.arange(nc_el2):
			print('step %i' %(i + nc_el1 ))
			j = int(i*float(steps)/nc_el2)
			atoms = extxyz.read_extxyz(filename, index = j)
			atoms = next(atoms)
			nl.build(atoms)
			cell = atoms.get_cell()
			ind_atom = int(i%len(elplace[1,0]))

			d = np.array([atoms.arrays['positions'][elplace[1,0]][ind_atom]])
			errvalue = 0
			try:
				force = atoms.get_array('force')[elplace[1,0]][ind_atom]   
			except KeyError:
				print('Forces in the xyz file are not present, or are not called force')
				force = None
				errvalue += 1
			try:
				energy = atoms.get_array('energy')[elplace[1,0]][ind_atom]
			except KeyError:
				print('Energies in the xyz file are not present, or are not called energy')
				energy = None
				errvalue += 1
				
			if errvalue ==2:
				print('Cannot find energy or force values in the xyz file, shutting down now')
				quit()
				
			indices, offsets = nl.get_neighbors(ind_atom)
			offsets = np.dot(offsets, cell)
			conf = np.zeros((len(indices), 5))
				
			for k, (a2, offset) in enumerate(zip(indices, offsets)):
				d = atoms.positions[a2] + offset - atoms.positions[elplace[1,0]][ind_atom]
				conf[k,:3] = d
				conf[k,4] = atoms.get_atomic_numbers()[a2]

			conf[:,3] = atoms.get_atomic_numbers()[elplace[1,0]][ind_atom]
			confs.append(conf)
			forces.append(force)
			energies.append(energy)
		
	else: 
		print("There is 1 element in the XYZ file")

		### Choose the number of entries each element will have, proportional to the square root of the ratio of occurrences of each element ###
		nc_el1 = n_data

		
		cutoffs = np.ones(N)*r_cut/2.         
		nl = NeighborList(cutoffs, skin=0., sorted=False, self_interaction=False,bothways=True)

		### Build conf and forces database centered on element 1 ###
		for i in np.arange(nc_el1):
			print('step %i' %(i))
			j = int(i*float(steps)/nc_el1)
			atoms = extxyz.read_extxyz(filename, index = j)
			atoms = next(atoms)
			nl.build(atoms)
			cell = atoms.get_cell()
			ind_atom = int(i%N)		# Select the atom number rotationg between the total atoms of el 1 present #

			d = np.array([atoms.arrays['positions'][ind_atom]])
			errvalue = 0
			try:
				force = atoms.get_array('force')[ind_atom]   
			except KeyError:
				print('Forces in the xyz file are not present, or are not called force')
				force = None
				errvalue+=1
			try:
				energy = atoms.get_array('energy')[ind_atom]
			except KeyError:
				print('Energies in the xyz file are not present, or are not called energy')
				energy = None
				errvalue+=1

			if errvalue ==2:
				print('Cannot find energy or force values in the xyz file, shutting down now')
				quit()
				
			indices, offsets = nl.get_neighbors(ind_atom)
			offsets = np.dot(offsets, cell)
			conf = np.zeros((len(indices), 5))
				
			for k, (a2, offset) in enumerate(zip(indices, offsets)):
				d = atoms.positions[a2] + offset - atoms.positions[ind_atom]
				conf[k,:3] = d
				conf[k,4] = atoms.get_atomic_numbers()[a2]					# Set the last digit of confs to be the element of the atom in the conf

			conf[:,3] = atoms.get_atomic_numbers()[ind_atom]  # Set the fourth digit of confs to be the element of the central atom
			confs.append(conf)
			forces.append(force)
			energies.append(energy)
			
	forces = np.array(forces)
	energies = np.array(energies)
	
	np.save("confs_cut=%.2f.npy" %(r_cut), confs)
	np.save("forces_cut=%.2f.npy"%(r_cut), forces)
	np.save("energies__cut=%.2f.npy"%(r_cut), energies)
	lens = []
	for i in np.arange(len(confs)):
		lens.append(len(confs[i]))

	print(max(lens))
	print(min(lens))
	print(np.mean(lens))
	
	return elementslist
#carve_confs('movie.xyz', 4.31)
