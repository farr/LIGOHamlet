import numpy as np
import numpy.random as nr

def draw_bg(size=(1,), snr_min=5.5, snr_std=2):
    bgs = nr.normal(loc=snr_min, scale=snr_std, size=size)

    sel = bgs < snr_min
    while np.count_nonzero(sel) > 0:
        bgs[sel] = nr.normal(loc=snr_min, scale=snr_std, size=np.count_nonzero(sel))
        sel = bgs < snr_min

    return bgs

def draw_fg(size=(1,), snr_min=5.5):
    us = nr.uniform(size=size)

    return snr_min/(1 - us)**(1.0/3.0)

def data_set(ncoinc, ffore, nbgs):
    bgs = [draw_bg(size=nb) for nb in nbgs]
    nf = int(round(ncoinc*ffore))
    fgs = [np.concatenate((draw_bg(ncoinc-nf), draw_fg(nf))) for nb in nbgs]

    return np.array(fgs), bgs
    
    
