import numpy as np

if __name__ == '__main__':
    directory = 'data/Fe_vac/'

    filename = 'MFF_2b_ntr_10_sig_1.00_cut_4.45.npy'
    arrays = np.load(directory + filename)
    np.save(directory+'MFF_2b_mapped_new.npy', arrays[3])

    filename = 'MFF_3b_ntr_10_sig_1.00_cut_4.45.npy'
    arrays = np.load(directory + filename)
    np.save(directory+'MFF_3b_mapped_new.npy', arrays[4])
