#!/usr/bin/env python

import argparse
import bz2
import glob
import numpy as np
import os.path as op
import plotutils.autocorr as ac

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', default='.', help='containing directory')

    args = parser.parse_args()

    for chainfile in glob.glob(op.join(args.dir, '*', 'chain.npy.bz2')):
        try:
            inp = bz2.BZ2File(chainfile, 'r')
            try:
                chain = np.load(inp)
            finally:
                inp.close()

            tchain = ac.emcee_thinned_chain(chain)

            out = bz2.BZ2File(op.join(op.dirname(chainfile), 'tchain.npy.bz2'), 'w')
            try:
                np.save(out, tchain)
            finally:
                out.close()
        except:
            print 'Could not complete thinning for ', chainfile
            continue

        print 'Thinned ', chainfile
