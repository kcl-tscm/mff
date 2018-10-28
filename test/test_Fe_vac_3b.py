import logging
import numpy as np

from ase.io import read
from pathlib import Path

from mff.interpolation import Spline3D
from mff.calculators import ThreeBodySingleSpecies

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    directory = Path('data/Fe_vac')

    print('========== Load trajectory ==========')

    filename = directory / 'movie.xyz'
    traj = read(str(filename), index=slice(0, 5))

    print('========== ThreeBodySingleSpecies ==========')

    # future: TwoBodySingleSpecies.from_json(directory / 'test.json')
    rs, element1, _, _, grid_data_1_1_1 = np.load(str(directory / 'MFF_3b_ntr_20_sig_1.00_cut_4.45.npy'))
    grid_1_1_1 = Spline3D(rs, rs, rs, grid_data_1_1_1)

    calc = ThreeBodySingleSpecies(r_cut=3.7, grid_1_1_1=grid_1_1_1)

    atoms = traj[0]
    atoms.set_calculator(calc)

    rms = np.sqrt(np.sum(np.square(atoms.arrays['force'] - atoms.get_forces()), axis=1))
    print('MAEF on forces: {:.4f} +- {:.4f}'.format(np.mean(rms), np.std(rms)))
    print(atoms.get_potential_energy())

    print('========== Calculate MAEF for each steps ==========')

    for atoms in traj:
        atoms.set_calculator(calc)

        rms = np.sqrt(np.sum(np.square(atoms.arrays['force'] - atoms.get_forces()), axis=1))
        print('MAEF on forces: {:.4f} +- {:.4f}'.format(np.mean(rms), np.std(rms)))
        print('energy: {}'.format(atoms.get_potential_energy()))

    # n_train = 10
    # ========== ThreeBodySingleSpecies ==========
    # INFO:mff.calculators:numbers is in system_changes
    # INFO:mff.calculators:initialize
    # MAEF on forces: 20365.7662 +- 9385.6598
    # -3056659.000264844
    # ========== Calculate MAEF for each steps ==========
    # MAEF on forces: 20365.7662 +- 9385.6598
    # energy: -3056659.000264844
    # MAEF on forces: 20352.8965 +- 9317.8726
    # energy: -3054539.971189826
    # MAEF on forces: 20307.5909 +- 9229.8828
    # energy: -3054210.657407379
    # MAEF on forces: 20261.9826 +- 9146.5965
    # energy: -3053729.815328974
    # MAEF on forces: 20235.5124 +- 9049.7248
    # energy: -3053171.188406065


    # n_train = 20
    # ========== ThreeBodySingleSpecies ==========
    # INFO:mff.calculators:numbers is in system_changes
    # INFO:mff.calculators:initialize
    # MAEF on forces: 82460.7472 +- 35603.8817
    # 2905106.355425914
    # ========== Calculate MAEF for each steps ==========
    # MAEF on forces: 82460.7472 +- 35603.8817
    # energy: 2905106.355425914
    # MAEF on forces: 81603.0771 +- 38472.0366
    # energy: 2916482.086216846
    # MAEF on forces: 81517.2225 +- 38519.9173
    # energy: 2913256.3958221003
    # MAEF on forces: 81418.0889 +- 38584.2406
    # energy: 2910011.553464438
    # MAEF on forces: 81310.2322 +- 38647.8092
    # energy: 2906773.0471959733