import logging
import numpy as np

from ase.io import read
from pathlib import Path

from mff.configurations import SamplingLinspaceSingle
from mff.interpolation import Spline1D
from mff.calculators import TwoBodySingleSpecies

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    directory = Path('data/Fe_vac')

    # Parameters:
    r_cut = 4.45
    n_bodies = 3

    n_train = 10
    n_test = 10
    sigma = 1.0
    noise = 0.00001

    grid_start = 1.5
    grid_spacing = 0.1
    grid_num = int(1 + (r_cut - grid_start) / grid_spacing)

    elementslist = [26]

    n_data = n_test + n_test

    print('========== Load trajectory ==========')

    filename = directory / 'movie.xyz'
    traj = read(str(filename), index=slice(0, 10))
