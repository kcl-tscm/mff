import numpy as np

if __name__ == '__main__':
    directory = 'data/HNi/'

    filename = 'MFF_3b_ntr_10_sig_1.00_cut_4.50_new.npy'
    arrays = np.load(directory + filename)
    # np.save(directory+'MFF_2b_mapped_new.npy', arrays[3])

    filename = 'MFF_3b_ntr_10_sig_1.00_cut_4.50.npy'
    arrays2 = np.load(directory + filename)
    # np.save(directory+'MFF_3b_mapped_new.npy', arrays[4])

# arrays[6][np.nonzero(arrays[6])]
# Out[12]:
# array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
# arrays[7][np.nonzero(arrays[7])]
# Out[13]:
# array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
#        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
# arrays[8][np.nonzero(arrays[8])]
# Out[14]:
# array([-2.36328274, -2.4418558 , -2.50408321, ...,  9.32230683,
#        10.00274446, 10.53414387])
# arrays[9][np.nonzero(arrays[9])]
# Out[15]:
# array([  3.27820747,   4.03157492,   4.83282734, ..., -15.05250635,
#        -14.27656349, -13.56952724])
