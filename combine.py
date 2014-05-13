#!/usr/bin/env python

import argparse
import bz2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import plotutils.plotutils as pu
import post_process as pp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', default='.', help='directory to scan for output subdirs')

    args = parser.parse_args()

    log_bfs = []
    plt.figure(1)
    plt.figure(2)
    for subdir in glob.glob(op.join(args.dir, '*')):
        if op.isdir(subdir):
            try:
                inp = bz2.BZ2File(op.join(subdir, 'tchain.npy.bz2'), 'r')
                try:
                    tchain = np.load(inp)
                finally:
                    inp.close()

                plt.figure(1)
                pu.plot_histogram_posterior(tchain[:,:,0].flatten(), normed=True, histtype='step')

                plt.figure(2)
                pu.plot_histogram_posterior(tchain[:,:,1].flatten(), normed=True, histtype='step')

                log_bfs.append(pp.log_odds_signal(tchain))
            except:
                print 'Couldn\'t process ', subdir
                continue


    plt.figure(1)
    plt.xlabel(r'$R_f$')
    plt.ylabel(r'$p\left( R_f \right)$')
    plt.savefig(op.join(args.dir, 'fore.pdf'))

    plt.figure(2)
    plt.xlabel(r'$R_b$')
    plt.ylabel(r'$p\left( R_b \right)$')
    plt.savefig(op.join(args.dir, 'back.pdf'))

    plt.figure()
    plt.plot(log_bfs, '*k')
    plt.ylabel(r'$\ln B_{s,n}$')
    plt.savefig(op.join(args.dir, 'bf.pdf'))
