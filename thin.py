#!/usr/bin/env python

import argparse
import bz2
import glob
import numpy as np
import os
import os.path as op
import plotutils.autocorr as ac
import sys
import tempfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--globdir', default='*', help='glob pattern for search directories')

    args = parser.parse_args()

    for chainfile in glob.glob(op.join(args.globdir, 'chain.npy.bz2')):
        dir = op.dirname(chainfile)
        tchainfile = op.join(dir, 'tchain.npy.bz2')
        temptchainfile = op.join(dir, 'temp_tchain.npy.bz2')
        
        if not op.isfile(tchainfile) or op.getmtime(chainfile) > op.getmtime(tchainfile):
            try:
                inp = bz2.BZ2File(chainfile, 'r')
                try:
                    chain = np.load(inp)
                finally:
                    inp.close()

                tchain = ac.emcee_thinned_chain(chain)
                tempout = bz2.BZ2File(temptchainfile, 'w')
                try:
                    np.save(tempout, tchain)
                    os.rename(temptchainfile, tchainfile)
                finally:
                    tempout.close()
            except:
                print 'Could not thin ', chainfile
                continue
            print 'Thinned ', chainfile
        else:
            print 'Skipped ', chainfile, ' due to previous thinning'
