import logging
import random
from abc import ABCMeta, abstractmethod

USE_ASAP = True
logger = logging.getLogger(__name__)


class Sampling(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass


class SingleSpecies(Sampling, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        super().__init__()


class TwoSpecies(Sampling, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        super().__init__()


class LinspaceSamplingSingle(SingleSpecies):
    def __init__(self, traj, n_target):
        super().__init__()

        self.traj = traj
        self.n_target = n_target

    def __iter__(self):

        n_element = 0
        for atoms in self.traj:
            n_element += len(atoms)

        if self.n_target > n_element:
            logger.warning('using all of them but ...')

            for atoms in self.traj:
                yield atoms, None

        else:
            step_size = n_element // self.n_target
            logger.info('step_size: {}'.format(step_size))

            current_ind = 0
            for atoms in self.traj:

                n_atoms = len(atoms)

                if current_ind >= n_atoms:
                    current_ind -= n_atoms
                    continue

                local_inds = range(current_ind, n_atoms, step_size)

                yield atoms, local_inds

                current_ind = current_ind + step_size * len(local_inds) - n_atoms

            # # Get the atomic number of each atom in the trajectory file
            # atom_number_list = [atoms.get_atomic_numbers() for atoms in traj]
            # flat_atom_number = np.concatenate(atom_number_list)
            # elements, elements_count = np.unique(flat_atom_number, return_counts=True)


class RandomSamplingSingle(SingleSpecies):
    def __init__(self, traj, n_target):
        super().__init__()

        self.traj = traj
        self.n_target = n_target

    def __iter__(self):
        # https://www.reddit.com/r/learnpython/comments/3sk1xj/splitting_a_list_in_sublists_by_values/

        n_element_list = [len(atoms) for atoms in self.traj]
        n_element = sum(n_element_list)

        if self.n_target > n_element:
            logger.warning('using all of them but ...')

            for atoms in self.traj:
                yield atoms, None

        else:

            random_inds = sorted(random.sample(range(n_element), self.n_target))

            def split_list(iterable, splitters):
                from itertools import accumulate

                splitters_iter = accumulate(splitters)
                local_inds, base = [], 0

                thresh = next(splitters_iter, None)
                for index in iterable:

                    while thresh is not None and index >= thresh:
                        yield local_inds
                        local_inds, base = [], thresh
                        thresh = next(splitters_iter, None)

                    local_inds.append(index - base)
                else:
                    yield local_inds

            logger.info('random sequence: {}'.format(random_inds))

            # from itertools import accumulate
            #
            # splitters_iter = accumulate(n_element_list)
            # local_inds, base = [], 0
            #
            # thresh = next(splitters_iter, None)
            # atoms = iter(self.traj)
            #
            # for index in random_inds:
            #
            #     while thresh is not None and index >= thresh:
            #         yield local_indsa
            #         local_inds, base = [], thresh
            #         thresh = next(splitters_iter, None)
            #
            #     local_inds.append(index - base)
            # else:
            #     yield local_inds

            local_inds_iter = split_list(random_inds, n_element_list)
            for atoms, local_inds in zip(self.traj, local_inds_iter):
                if local_inds:
                    yield atoms, local_inds


if __name__ == '__main__':
    import numpy as np

    logging.basicConfig(level=logging.INFO)

    test_data = np.arange(100).reshape(10, -1)

    for data, local_inds in RandomSamplingSingle(test_data, 10):
        print(data, local_inds)
