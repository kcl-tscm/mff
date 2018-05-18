import numpy as np
from abc import ABCMeta, abstractmethod


# TODO: move the grid builders to the GP class
# TODO: combining grid with spline1D and spline3D
# TODO: missing save and load from file functions
# TODO: missing feature: nnodes and save astype(dtype=np.dtype('f4'))

def generate_triplets(dists):
    d_ij, d_jk, d_ki = np.meshgrid(dists, dists, dists, indexing='ij', sparse=False, copy=True)

    # Valid triangles according to triangle inequality
    inds = np.logical_and(d_ij <= d_jk + d_ki, np.logical_and(d_jk <= d_ki + d_ij, d_ki <= d_ij + d_jk))

    # Utilizing permutation invariance
    inds = np.logical_and(np.logical_and(d_ij >= d_jk, d_jk >= d_ki), inds)

    # Element on the x axis
    r_ij_x = d_ij[inds]

    # Element on the xy plane
    r_ki_x = (d_ij[inds] ** 2 - d_jk[inds] ** 2 + d_ki[inds] ** 2) / (2 * d_ij[inds])

    # using abs to avoid numerical error near to 0
    r_ki_y = np.sqrt(np.abs(d_ki[inds] ** 2 - r_ki_x ** 2))

    return inds, r_ij_x, r_ki_x, r_ki_y


class Grid(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, gp, start, stop, num):
        self.gp = gp
        self.start = start
        self.stop = stop
        self.num = num


class SingleSpeciesGrid(Grid):
    def __init__(self, gp, start, stop, num, element1, build2donly=True):
        super().__init__(gp, start, stop, num)
        self.element1 = element1

        self.grid_2b = None
        self.grid_3b = None
        self.build2donly = build2donly

    def build_grids(self):
        dists = np.linspace(self.start, self.stop, self.num)

        self.build_2_grid(dists)
        if not self.build2donly:
            self.build_3_grid(dists)

        return dists, self.element1, self.element1, self.grid_2b, self.grid_3b

    def build_2_grid(self, dists):
        """Function that builds and predicts energies on a line of values"""

        confs = np.zeros((self.num, 1, 5))

        confs[:, 0, 0] = dists
        confs[:, 0, 3], confs[:, 0, 4] = self.element1, self.element1

        self.grid_2b = self.gp.predict_energy(confs)

    def build_3_grid(self, dists):
        """Function that builds and predicts energies on a cube of values"""

        inds, r_ij_x, r_ki_x, r_ki_y = generate_triplets(dists)

        confs = np.zeros((len(r_ij_x), 2, 5))

        confs[:, 0, 0] = r_ij_x  # Element on the x axis
        confs[:, 1, 0] = r_ki_x  # Reshape into confs shape: this is x2
        confs[:, 1, 1] = r_ki_y  # Reshape into confs shape: this is y2

        # Permutations of elements

        confs[:, :, 3] = self.element1  # Central element is always element 1
        confs[:, 0, 4] = self.element1  # Element on the x axis is always element 2
        confs[:, 1, 4] = self.element1  # Element on the xy plane is always element 3

        self.grid_3b = np.zeros((self.num, self.num, self.num))
        self.grid_3b[inds] = self.gp.predict_energy(confs).flatten()

        for ind_i in range(self.num):
            for ind_j in range(ind_i + 1):
                for ind_k in range(ind_j + 1):
                    self.grid_3b[ind_i, ind_k, ind_j] = self.grid_3b[ind_i, ind_j, ind_k]
                    self.grid_3b[ind_j, ind_i, ind_k] = self.grid_3b[ind_i, ind_j, ind_k]
                    self.grid_3b[ind_j, ind_k, ind_i] = self.grid_3b[ind_i, ind_j, ind_k]
                    self.grid_3b[ind_k, ind_i, ind_j] = self.grid_3b[ind_i, ind_j, ind_k]
                    self.grid_3b[ind_k, ind_j, ind_i] = self.grid_3b[ind_i, ind_j, ind_k]


class TwoSpeciesGrid(Grid):
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

    def build_2_grid(self, dists):
        """Function that builds and predicts energies on a line of values"""

        confs = np.zeros((self.num, 1, 5))
        confs[:, 0, 0] = dists

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
