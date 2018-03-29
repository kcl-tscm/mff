import numpy as np

if __name__ == '__main__':
    directory = 'data/HNi/'

    filename = 'MFF_3b_ntr_10_sig_1.00_cut_4.50_new.npy'
    arrays = np.load(directory + filename)
    # np.save(directory+'MFF_2b_mapped_new.npy', arrays[3])

    filename = 'MFF_3b_ntr_10_sig_1.00_cut_4.50.npy'
    arrays2 = np.load(directory + filename)
    # np.save(directory+'MFF_3b_mapped_new.npy', arrays[4])

