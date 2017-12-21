'''
Created on 21 Dec 2017

@author: Saumitra
'''

import numpy as np
from skimage.segmentation import slic

def random_selection(norm_inv, random_bin = False, random_block = False, debug_print = False):
    
    n_bin_enabled = (norm_inv == 1).sum()
    n_bins = norm_inv.shape[0] * norm_inv.shape[1]
    area_enabled = n_bin_enabled/float(n_bins)
    if debug_print:
        print(" Percentage of the bins enabled: %.2f" %(100 * area_enabled))
    
    if random_bin:
        # randomisation: we randomise the bins not the blocks: Not sure if this a good approach
        random_bin_idx = np.random.choice(range(n_bins), n_bin_enabled, replace = False)
        norm_inv_flat = np.ravel(np.zeros((norm_inv.shape[0], norm_inv.shape[1])))
        for idx in random_bin_idx:
            norm_inv_flat[idx] = 1
        norm_inv = norm_inv_flat.reshape(norm_inv[0].shape[0], norm_inv[0].shape[1])
    elif random_block:
        # keep the area covered nearly similar and use it to select N blocks randomly from a total of M
        M = 24 # inspired from SLIME for ease of implementation, i.e. divide 115 x 80 matrix into 24 segments
        N = int(np.floor(area_enabled * M))  # kept floor just to be a little miser on the blocks enabled
        segments = slic(norm_inv, n_segments= M, compactness=1)
        n_features = np.unique(segments).shape[0]
        ones = np.random.choice(range(M), N, replace = False)    # non-repetitive random integers, selects indices randomly that are to be masked
        if debug_print:
            print("Number of blocks enabled: %d" %N)
            print("Number of segments: %d" %(n_features))
            print("Enabled segments")
            print(ones)
        norm_inv = np.zeros(segments.shape).astype(bool)
        for z in ones:
                norm_inv[segments == z] = True
    else: # no randomisation
        pass
    return norm_inv, area_enabled
        
        
        
        