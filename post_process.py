#!/usr/bin/env python

import argparse
import bz2
import matplotlib.pyplot as pp
import numpy as np
import os.path as op
import pickle
import plotutils.autocorr as ac
import plotutils.plotutils as pu
import posterior as pos
import scipy.stats as ss
import triangle

def triangle_plot(chain, outdir):
    fchain = chain.reshape((-1, chain.shape[2]))

    triangle.corner(fchain[:,:2], labels=[r'$R_f$', r'$R_b$'], quantiles=[0.05, 0.95])

    pp.savefig(op.join(outdir, 'rates.pdf'))

def pback_plot(chain, logpost, outdir):
    pbacks = []
    for p in chain.reshape((-1, chain.shape[2])):
        pbacks.append(logpost.pbacks(p))
    pbacks = np.array(pbacks)

    mean_pback = np.mean(pbacks, axis=0)
    low_pback = np.percentile(pbacks, 31.8, axis=0)
    high_pback = np.percentile(pbacks, 68.2, axis=0)

    dplow = low_pback - mean_pback
    dphigh = high_pback - mean_pback

    rhos = np.sqrt(np.sum(logpost.coincs*logpost.coincs, axis=1))

    pp.errorbar(rhos, mean_pback, yerr=np.array([dplow, dphigh]), fmt='.', color='k')

    pp.axis(ymax=2, ymin=np.min(low_pback)/2.0)

    pp.yscale('log')

    pp.xlabel(r'$\rho$')
    pp.ylabel(r'$p(\mathrm{background})$')

    pp.savefig(op.join(outdir, 'pbacks.pdf'))

def log_odds_signal(tchain, rhigh=100.0):
    kde = ss.gaussian_kde(np.sqrt(tchain[:,:,0].flatten()/rhigh))

    return kde(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', default='.', help='directory for chain files')

    args = parser.parse_args()
    
    pp.rcParams['text.usetex'] = True

    inp = bz2.BZ2File(op.join(args.dir, 'chain.npy.bz2'), 'r')
    try:
        chain = np.load(inp)
    finally:
        inp.close()

    tchain = ac.emcee_thinned_chain(chain)

    if tchain is None:
        print 'Could not post-process'
        exit(1)

    inp = bz2.BZ2File(op.join(args.dir, 'posterior.pkl.bz2'), 'r')
    try:
        oldlogpost = pickle.load(inp)
    finally:
        inp.close()
    logpost = pos.Posterior(oldlogpost.coincs, oldlogpost.bgs, oldlogpost.snr_min, foreground=oldlogpost.foreground)

    triangle_plot(tchain, args.dir)
    pback_plot(tchain, logpost, args.dir)
