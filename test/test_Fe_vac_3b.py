import logging
import numpy as np

from ase.io import read
from pathlib import Path

from m_ff.calculators import Spline3D, ThreeBodySingleSpecies

logging.basicConfig(level=logging.INFO)

# import matplotlib.pyplot as plt

if __name__ == '__main__':

    directory = Path('data/Fe_vac')

    print('========== Load trajectory ==========')

    filename = directory / 'movie.xyz'
    traj = read(str(filename), index=slice(None))

    print('========== ThreeBodySingleSpecies ==========')

    # future: TwoBodySingleSpecies.from_json(directory / 'test.json')
    rs, element1, _, _, grid_1_1_1 = np.load(str(directory / 'MFF_3b_ntr_10_sig_1.00_cut_4.45.npy'))
    grid_1_1_1 = Spline3D(rs, rs, rs, grid_1_1_1)

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


    # ========== ThreeBodySingleSpecies ==========
    # INFO:m_ff.calculators:numbers is in system_changes
    # INFO:m_ff.calculators:initialize
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
    # MAEF on forces: 20149.0098 +- 8953.6164
    # energy: -3050802.056027134
    # MAEF on forces: 19884.9344 +- 9051.7896
    # energy: -3047407.5538894148
    # MAEF on forces: 20029.8935 +- 9021.6495
    # energy: -3047401.4206132893
    # MAEF on forces: 20417.3964 +- 9293.7082
    # energy: -3046087.8447186737
    # MAEF on forces: 20432.0254 +- 9233.4040
    # energy: -3048276.7083526854
    # MAEF on forces: 20145.3694 +- 9630.9514
    # energy: -3046065.2009255127
    # MAEF on forces: 20353.6620 +- 9702.8628
    # energy: -3046309.3593773562
    # MAEF on forces: 20393.6307 +- 9777.6315
    # energy: -3045579.5074405354
    # MAEF on forces: 20293.9386 +- 9817.1483
    # energy: -3045418.029207301
    # MAEF on forces: 20355.6648 +- 9942.7738
    # energy: -3045102.104442049
    # MAEF on forces: 20362.8833 +- 10245.6027
    # energy: -3043719.014150446
    # MAEF on forces: 20334.1289 +- 10410.4893
    # energy: -3042521.5239588064
    # MAEF on forces: 20375.6755 +- 10533.0689
    # energy: -3042331.3455765937
    # MAEF on forces: 20447.0275 +- 10545.5746
    # energy: -3040821.6634461335
    # MAEF on forces: 20354.2184 +- 10726.4408
    # energy: -3040451.331337367
    # MAEF on forces: 20407.7554 +- 10741.6328
    # energy: -3040817.4606952984
    # MAEF on forces: 20648.6367 +- 10494.2623
    # energy: -3042867.405653011
    # MAEF on forces: 20695.5133 +- 10496.0712
    # energy: -3043547.5234750975
    # MAEF on forces: 20726.2948 +- 10492.3889
    # energy: -3044327.107237488
    # MAEF on forces: 20591.4169 +- 10448.8937

