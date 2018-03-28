import numpy as np
from abc import ABCMeta, abstractmethod


class Grids_building(object):

    def __init__(self, d0, df, nr, gp, element1, element2=None, nnodes=1):
        self.d0 = d0  # Grid starting point
        self.df = df  # Grid ending point
        self.nr = nr  # Number of grid points
        self.element1 = element1  # Number of element 1
        self.element2 = element2  # Number of element 2, if present
        self.gp = gp
        self.nnodes = nnodes

    ### Function used to go from a r1 r2 r3 form to a x1 x2 y2 form of the grid of values ###
    @staticmethod
    def from_r_to_xy(a):
        shape = np.shape(a)
        b = np.zeros(shape)
        b[:, :, :, 0] = a[:, :, :, 0]  # x1
        b[:, :, :, 1] = (np.square(a[:, :, :, 0]) + np.square(a[:, :, :, 1]) - np.square(a[:, :, :, 2])) / (
                2.0 * a[:, :, :, 0])  # x2
        b[:, :, :, 2] = a[:, :, :, 1] * np.sin(np.arccos(
            (np.square(a[:, :, :, 0]) + np.square(a[:, :, :, 1]) - np.square(a[:, :, :, 2])) / (
                    2.0 * a[:, :, :, 0] * a[:, :, :, 1])))  # y2

        for i in np.arange(len(b[:, 0, 0, 0])):  # Do this to take away triplets that are not physical
            for j in np.arange(len(b[0, :, 0, 0])):
                for k in np.arange(len(b[0, 0, :, 0])):
                    if np.isnan(b[i, j, k, 2]):
                        b[i, j, k, :] = np.zeros(3)
        return (b)

    ### Function that calls the building of all of the appropriate grids, depending on whether there is a single element or two ###
    def build_grids(self):
        d0 = self.d0
        df = self.df
        nr = self.nr
        element1 = self.element1
        element2 = self.element2
        rs = np.linspace(d0, df, nr)

        if element2 == None:  # If single element, build only a 2- and  3-body grid
            grid_1_1 = self.build_2_grid(d0, df, nr, element1, element1)
            grid_1_1_1 = self.build_3_grid(d0, df, nr, element1, element1, element1)
            result = [rs, element1, element2, grid_1_1, grid_1_1_1]

        else:  # If single element, build three 2- and four 3-body grids
            grid_1_1 = self.build_2_grid(d0, df, nr, element1, element1)
            grid_1_2 = self.build_2_grid(d0, df, nr, element1, element2)
            grid_2_2 = self.build_2_grid(d0, df, nr, element2, element2)

            grid_1_1_1 = self.build_3_grid(d0, df, nr, element1, element1, element1)
            grid_1_1_2 = self.build_3_grid(d0, df, nr, element1, element1, element2)
            grid_1_2_2 = self.build_3_grid(d0, df, nr, element1, element2, element2)
            grid_2_2_2 = self.build_3_grid(d0, df, nr, element2, element2, element2)

            result = [rs, element1, element2, grid_1_1, grid_1_2, grid_2_2, grid_1_1_1, grid_1_1_2, grid_1_2_2,
                      grid_2_2_2]

        return result

    ### Function that builds and ppredicts energies on a line of values ###
    def build_2_grid(self, d0, df, nr, el_1, el_2):
        gp = self.gp
        rs = np.linspace(d0, df, nr)
        confs = np.zeros((nr, 1, 5))
        confs[:, 0, 0] = rs
        confs[:, 0, 3] = el_1
        confs[:, 0, 4] = el_2
        energies = gp.predict_energy(confs)

        energies = energies.astype(dtype=np.dtype('f4'))  # Reduce precision on the energy grid
        return (energies)

    ### Function that builds and predicts energies on a cube of values ###
    def build_3_grid(self, d0, df, nr, el_1, el_2, el_3):
        gp = self.gp
        nnodes = self.nnodes
        rs = np.linspace(d0, df, nr)
        confs = np.zeros((nr * nr * nr, 2, 5))

        confs[:, :, 3] = el_1  # Central element is always element 1
        confs[:, 0, 4] = el_2  # Element on the x axis is always element 2
        confs[:, 1, 4] = el_3  # Element on the xy plane is always element 3

        coords = np.zeros((nr, nr, nr, 3))  # Build a coordinates grid that spans r1 r2 r3 values
        coords[:, :, :, 0] = rs[:, None, None]
        coords[:, :, :, 1] = rs[None, :, None]
        coords[:, :, :, 2] = rs[None, None, :]

        badgrid = self.from_r_to_xy(coords)  # Convert the coords into a x1 x2 y2 grid

        confs[:, 0, 0] = np.reshape(badgrid[:, :, :, 0], nr ** 3)  # Reshape into confs shape: this is x1
        confs[:, 1, 0] = np.reshape(badgrid[:, :, :, 1], nr ** 3)  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = np.reshape(badgrid[:, :, :, 2], nr ** 3)  # Reshape into confs shape: this is y2

        if nnodes > 1:
            from pathos.multiprocessing import ProcessingPool  # Import multiprocessing package
            n = len(confs)

            if __name__ == 'create_MFF_grid':
                print('Using %i cores for the remapping' % (nnodes))
                pool = ProcessingPool(nodes=nnodes)
                splitind = np.zeros(nnodes + 1)
                factor = (n + (nnodes - 1)) / nnodes
                splitind[1:-1] = [(i + 1) * factor for i in np.arange(nnodes - 1)]
                splitind[-1] = n
                splitind = splitind.astype(int)
                clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nnodes)]
                result = np.array(pool.map(gp.predict_energy, clist))
                result = np.concatenate(result)
            all_energies = result

        else:
            all_energies = gp.predict_energy(confs)

        all_energies = all_energies.astype(dtype=np.dtype('f4'))  # Reduce precision on the energy grid
        return all_energies


# TODO: move the grid builders to the GP class
# TODO: combining grid with spline1D and spline3D
# TODO: missing save and load from file functions
# TODO: missing feature: nnodes and save astype(dtype=np.dtype('f4'))

class Grid(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, gp, start, stop, num):
        self.gp = gp
        self.start = start
        self.stop = stop
        self.num = num

    @staticmethod
    def generate_triplets(dists):
        d_ij, d_jk, d_ki = np.meshgrid(dists, dists, dists, indexing='ij', sparse=False, copy=True)

        # Valid triangles according to triangle inequality
        inds = np.logical_and(d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

        # Element on the x axis
        r1_x = d_ij[inds].ravel()

        # Element on the xy plane
        r2_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 + d_ki[inds] ** 2) / (2 * d_ij[inds])
        r2_y = np.sqrt(d_ki[inds] ** 2 - r2_x ** 2)

        return inds, r1_x, r2_x, r2_y


class SingleScecies(Grid):
    def __init__(self, gp, start, stop, num, element1):
        super().__init__(gp, start, stop, num)
        self.element1 = element1

        self.grid_1_1 = None
        self.grid_1_1_1 = None

    def build_grids(self):
        dists = np.linspace(self.start, self.stop, self.num)

        self.build_2_grid(dists)
        self.build_3_grid(dists)

        return dists, self.element1, self.element1, self.grid_1_1, self.grid_1_1_1

    def build_2_grid(self, dists):
        """Function that builds and predicts energies on a line of values"""

        confs = np.zeros((self.num, 1, 5))

        confs[:, 0, 0] = dists
        confs[:, 0, 3], confs[:, 0, 4] = self.element1, self.element1

        self.grid_1_1 = self.gp.predict_energy(confs)

    def build_3_grid(self, dists):
        """Function that builds and predicts energies on a cube of values"""

        inds, r1_x, r2_x, r2_y = self.generate_triplets(dists)

        confs = np.zeros((np.sum(inds), 2, 5))

        confs[:, 0, 0] = r1_x  # Element on the x axis
        confs[:, 1, 0] = r2_x  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = r2_y  # Reshape into confs shape: this is y2

        # Permutations of elements

        confs[:, :, 3] = self.element1  # Central element is always element 1
        confs[:, 0, 4] = self.element1  # Element on the x axis is always element 2
        confs[:, 1, 4] = self.element1  # Element on the xy plane is always element 3

        self.grid_1_1_1 = np.zeros((self.num, self.num, self.num))
        self.grid_1_1_1[inds] = self.gp.predict_energy(confs).ravel()


class TwoSpecies(Grid):
    def __init__(self, gp, start, stop, num, element1, element2):
        super().__init__(gp, start, stop, num)
        self.element1 = element1
        self.element2 = element2

        self.grid_1_1 = None
        self.grid_1_2 = None
        self.grid_2_2 = None
        self.grid_1_1_1 = None
        self.grid_1_1_2 = None
        self.grid_1_2_2 = None
        self.grid_2_2_2 = None

    def build_grids(self):
        """Function that calls the building of all of the appropriate grids"""

        dists = np.linspace(self.start, self.stop, self.num)

        self.build_2_grid(dists)
        self.build_3_grid(dists)

        return (dists, self.element1, self.element2,
                self.grid_1_1, self.grid_1_2, self.grid_2_2,
                self.grid_1_1_1, self.grid_1_1_2, self.grid_1_2_2, self.grid_2_2_2)

    def build_2_grid(self, rs):
        """Function that builds and predicts energies on a line of values"""

        confs = np.zeros((len(rs), 1, 5))
        confs[:, 0, 0] = rs

        confs[:, 0, 3], confs[:, 0, 4] = self.element1, self.element1
        self.grid_1_1 = self.gp.predict_energy(confs)

        confs[:, 0, 3], confs[:, 0, 4] = self.element1, self.element2
        self.grid_1_2 = self.gp.predict_energy(confs)

        confs[:, 0, 3], confs[:, 0, 4] = self.element2, self.element2
        self.grid_2_2 = self.gp.predict_energy(confs)

    def build_3_grid(self, dists):
        """Function that builds and predicts energies on a cube of values"""

        inds, r1_x, r2_x, r2_y = self.generate_triplets(dists)

        confs = np.zeros((np.sum(inds), 2, 5))

        confs[:, 0, 0] = r1_x  # Element on the x axis
        confs[:, 1, 0] = r2_x  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = r2_y  # Reshape into confs shape: this is y2

        # Permutations of elements

        confs[:, :, 3] = self.element1  # Central element is always element 1
        confs[:, 0, 4] = self.element1  # Element on the x axis is always element 2
        confs[:, 1, 4] = self.element1  # Element on the xy plane is always element 3
        self.grid_1_1_1 = np.zeros((self.num, self.num, self.num))
        self.grid_1_1_1[inds] = self.gp.predict_energy(confs).ravel()

        confs[:, :, 3] = self.element1  # Central element is always element 1
        confs[:, 0, 4] = self.element1  # Element on the x axis is always element 2
        confs[:, 1, 4] = self.element2  # Element on the xy plane is always element 3
        self.grid_1_1_2 = np.zeros((self.num, self.num, self.num))
        self.grid_1_1_2[inds] = self.gp.predict_energy(confs).ravel()

        confs[:, :, 3] = self.element1  # Central element is always element 1
        confs[:, 0, 4] = self.element2  # Element on the x axis is always element 2
        confs[:, 1, 4] = self.element2  # Element on the xy plane is always element 3
        self.grid_1_2_2 = np.zeros((self.num, self.num, self.num))
        self.grid_1_2_2[inds] = self.gp.predict_energy(confs).ravel()

        confs[:, :, 3] = self.element2  # Central element is always element 1
        confs[:, 0, 4] = self.element2  # Element on the x axis is always element 2
        confs[:, 1, 4] = self.element2  # Element on the xy plane is always element 3
        self.grid_2_2_2 = np.zeros((self.num, self.num, self.num))
        self.grid_2_2_2[inds] = self.gp.predict_energy(confs).ravel()

        # energies = np.zeros((nr, nr, nr))
        # energies[inds] = gp.predict_energy(confs).ravel()
        # np.any(np.isnan(energies))
        # np.sum(inds)
        # len(energies[np.nonzero(energies)])
        # return energies.astype(dtype=np.dtype('f4'))  # Reduce precision on the energy grid
