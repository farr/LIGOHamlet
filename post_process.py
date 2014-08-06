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
    pp.savefig(op.join(outdir, 'rates.png'))


def pback_plot(chain, logpost, outdir):
    pbacks = []
    for p in chain.reshape((-1, chain.shape[2])):
        pbacks.append(logpost.pbacks(p))
    pbacks = np.array(pbacks)

    mean_pback = np.mean(pbacks, axis=0)
    sigma_pback = np.std(pbacks, axis=0)

    rhos = np.sqrt(np.sum(logpost.coincs*logpost.coincs, axis=1))

    pp.errorbar(rhos, mean_pback, yerr=sigma_pback, fmt='.', color='k')

    pp.yscale('log', nonposy='clip')

    pp.xlabel(r'$\rho$')
    pp.ylabel(r'$p(\mathrm{background})$')

    pp.savefig(op.join(outdir, 'pbacks.pdf'))
    pp.savefig(op.join(outdir, 'pbacks.png'))

def pback_dist_plot(chain, logpost, outdir):
    log_pbacks = []
    for p in chain.reshape((-1, chain.shape[2])):
        log_pbacks.append(logpost.log_pbacks(p))
    log_pbacks = np.array(log_pbacks)

    log_pbacks /= np.log(10.0)

    for lpb in log_pbacks:
        pu.plot_histogram_posterior(lpb, normed=True, histtype='step')

    pp.xlabel(r'$\log_{10}\left(p_\mathrm{back}\right)$')
    pp.ylabel(r'$p\left(\log_{10}\left(p_\mathrm{back}\right)\right)$')

    pp.savefig(op.join(outdir, 'pback-dist.pdf'))
    pp.savefig(op.join(outdir, 'pback-dist.png'))

def log_odds_signal(tchain, rhigh=100.0):
    kde = ss.gaussian_kde(np.sqrt(tchain[:,:,0].flatten()/rhigh))

    return -np.log(kde(0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', default='.', help='directory for chain files')

    args = parser.parse_args()
    
    pp.rcParams['text.usetex'] = True

    inp = bz2.BZ2File(op.join(args.dir, 'tchain.npy.bz2'), 'r')
    try:
        tchain = np.load(inp)
    finally:
        inp.close()

    inp = bz2.BZ2File(op.join(args.dir, 'posterior.pkl.bz2'), 'r')
    try:
        oldlogpost = pickle.load(inp)
    finally:
        inp.close()
    logpost = pos.Posterior(oldlogpost.coincs, oldlogpost.bgs, oldlogpost.snr_min, foreground=oldlogpost.foreground)

    triangle_plot(tchain, args.dir)
    pp.clf()
    pback_plot(tchain, logpost, args.dir)
    pback_dist_plot(tchain, logpost, args.dir)

    out = open(op.join(args.dir, 'bf.dat'), 'w')
    try:
        out.write('# log(BF)\n{0:g}\n'.format(log_odds_signal(tchain)[0]))
    finally:
        out.close()
