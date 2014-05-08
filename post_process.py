#!/usr/bin/env python

import argparse
import bz2
import matplotlib.pyplot as pp
import numpy as np
import plotutils.autocorr as ac
import plotutils.plotutils as pu
import posterior as pos
import triangle

def triangle_plot(chain):
    fchain = chain.reshape((-1, chain.shape[2]))

    triangle.corner(fchain[:,:2], labels=[r'$R_f$', r'$R_b$'], quantiles=[0.05, 0.95])

    pp.savefig('rates.pdf')

if __name__ == '__main__':
    pp.rcParams['text.usetex'] = True

    inp = bz2.BZ2File('chain.npy.bz2', 'r')
    try:
        chain = np.load(inp)
    finally:
        inp.close()

    tchain = ac.emcee_thinned_chain(chain)

    if tchain is None:
        print 'Could not post-process'
        exit(1)

    triangle_plot(tchain)
