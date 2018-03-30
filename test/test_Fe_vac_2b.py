import logging
import numpy as np

from ase.io import read
from pathlib import Path

from m_ff.interpolation import Spline1D
from m_ff.calculators import TwoBodySingleSpecies

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    directory = Path('data/Fe_vac')

    print('========== Load trajectory ==========')

    filename = directory / 'movie.xyz'
    traj = read(str(filename), index=slice(0, 5))

    print('========== TwoBodySingleSpecies ==========')

    # future: TwoBodySingleSpecies.from_json(directory / 'test.json')
    rs, element1, _, grid_data_1_1, _ = np.load(str(directory / 'MFF_2b_ntr_10_sig_1.00_cut_4.45.npy'))
    grid_1_1 = Spline1D(rs, grid_data_1_1)

    calc = TwoBodySingleSpecies(r_cut=3.7, grid_1_1=grid_1_1)

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

    # ========== TwoBodySingleSpecies ==========
    # INFO:m_ff.calculators:numbers is in system_changes
    # INFO:m_ff.calculators:initialize
    # MAEF on forces: 0.9893 +- 0.4852
    # MAEF on forces: 0.9893 +- 0.4852
    # MAEF on forces: 0.9742 +- 0.4746
    # MAEF on forces: 0.9606 +- 0.4629
    # MAEF on forces: 0.9491 +- 0.4495
    # MAEF on forces: 0.9418 +- 0.4332
    # MAEF on forces: 0.9360 +- 0.4174
    # MAEF on forces: 0.9315 +- 0.4055
    # MAEF on forces: 0.9292 +- 0.3950
    # MAEF on forces: 0.9304 +- 0.3832
    # MAEF on forces: 0.9333 +- 0.3762
    # MAEF on forces: 0.9363 +- 0.3752
    # MAEF on forces: 0.9431 +- 0.3784
    # MAEF on forces: 0.9537 +- 0.3835
    # MAEF on forces: 0.9646 +- 0.3929
    # MAEF on forces: 0.9734 +- 0.4055
    # MAEF on forces: 0.9836 +- 0.4158
    # MAEF on forces: 0.9981 +- 0.4195
    # MAEF on forces: 1.0130 +- 0.4237
    # MAEF on forces: 1.0232 +- 0.4296
    # MAEF on forces: 1.0338 +- 0.4352
    # MAEF on forces: 1.0390 +- 0.4412
    # MAEF on forces: 1.0401 +- 0.4469
    # MAEF on forces: 1.0384 +- 0.4530
    # MAEF on forces: 1.0344 +- 0.4558
    # MAEF on forces: 1.0267 +- 0.4612
    # MAEF on forces: 1.0190 +- 0.4648
    # MAEF on forces: 1.0101 +- 0.4676
