# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:44:09 2024

@author: marce
"""

import numpy as np
from skimage.morphology import disk
from scipy.ndimage import binary_dilation

def buffer_patches(patches: np.ndarray, radius_px=3, print_interval=200):
    assert len(patches.shape) == 3, 'Patches must be in shape (B, H, W)'
    
    # Build structuring element
    struct_elem = disk(radius_px)

    size = len(patches) # Total number of patches 
    result = [] # Result list
    
    for i, patch in enumerate(patches):
        if i % print_interval == 0:
            print(f'Buffering patch {i:>6d}/{size:>6d}')
            
        buffered_patch = binary_dilation(patch, struct_elem).astype(np.uint8)
        result.append(buffered_patch)
        
    result = np.array(result) # Aggregate list
    
    return result