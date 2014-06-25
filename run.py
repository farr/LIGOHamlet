#!/usr/bin/env python

import argparse
import bz2
import data as d
import emcee
import numpy as np
import os
import os.path as op
import pickle
import plotutils.autocorr as ac
import posterior as pos
import sys

def find_best_rate(p0, logpost):
    Nmax = logpost.coincs.shape[0]

    Rs = np.arange(0.5, Nmax+0.5)

    def f(r):
        p = p0.copy()
        p[0] = r
        p[1] = Nmax-r
        return logpost(p)

    fs = np.array([f(r) for r in Rs])
    imax = np.argmax(fs)

    return Rs[imax]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--coincs', required=True, help='coinc file')
    parser.add_argument('--trigs1', required=True, help='triggers file for IFO 1')
    parser.add_argument('--trigs2', required=True, help='triggers file for IFO 2')

    parser.add_argument('--snr-min', default=5.5, type=float, help='threshold SNR')
    parser.add_argument('--neff', default=10000, type=int, help='number of independent samples')
    parser.add_argument('--outdir', default='.', help='output directory')
    parser.add_argument('--nfore', default=10000000, type=int, help='number of foreground draws')
    parser.add_argument('--nwalkers', default=128, type=int, help='number of walkers')
    parser.add_argument('--nthin', default=100, type=int, help='thinning parameter')
    parser.add_argument('--nstep', default=10000, type=int, help='number of steps between testing ACL')

    args = parser.parse_args()

    nstep_thin = args.nstep / args.nthin

    ctrig, trig1, trig2 = d.process_coinc_triggers(args.coincs, args.trigs1, args.trigs2)
    
    coincs = np.column_stack((ctrig['SNR1'], ctrig['SNR2']))
    bgs = [trig1['SNR'], trig2['SNR']]

    try:
        inp = bz2.BZ2File(op.join(args.outdir, 'posterior.pkl.bz2'), 'r')
        try:
            logpost = pickle.load(inp)
            print 'Loaded pickled posterior'
            sys.__stdout__.flush()
        finally:
            inp.close()
    except:
        logpost = pos.Posterior(coincs, bgs, args.snr_min, N=args.nfore)
        print 'Generated new posterior'
        sys.__stdout__.flush()

    try:
        inp = open(op.join(args.outdir, 'thin.txt'), 'r')
        try:
            thin = int(inp.readline())
            print 'Loaded thin parameter = ', thin
            sys.__stdout__.flush()
        finally:
            inp.close()
    except:
        thin = args.nthin
        print 'Could not load thin parameter, using ', thin
        sys.__stdout__.flush()

    out = open(op.join(args.outdir, 'thin.txt'), 'w')
    try:
        out.write('{0:d}\n'.format(thin))
    finally:
        out.close()

    out = bz2.BZ2File(op.join(args.outdir, 'temp_posterior.pkl.bz2'), 'w')
    try:
        pickle.dump(logpost, out)
    finally:
        out.close()
    os.rename(op.join(args.outdir, 'temp_posterior.pkl.bz2'),
              op.join(args.outdir, 'posterior.pkl.bz2'))
    print 'Saved posterior to pickle.'
    sys.__stdout__.flush()

    sampler = emcee.EnsembleSampler(args.nwalkers, logpost.nparams, logpost)

    p0 = logpost.params_guess()

    rbest = find_best_rate(p0, logpost)
    p0[0] = rbest
    p0[1] = logpost.coincs.shape[0] - rbest

    ps = np.exp(np.log(p0) + 1e-3*np.random.randn(args.nwalkers, logpost.nparams))

    try:
        cin = bz2.BZ2File(op.join(args.outdir, 'chain.npy.bz2'), 'r')
        try:
            lin = bz2.BZ2File(op.join(args.outdir, 'lnprob.npy.bz2'), 'r')
            try:
                sampler._chain = np.load(cin)
                sampler._lnprob = np.load(lin)

                print 'Loaded old chain, size = ', sampler.chain.shape[1], ' steps'
                sys.__stdout__.flush()
            finally:
                lin.close()
        finally:
            cin.close()
    except:
        print 'Starting fresh chain.'
        sys.__stdout__.flush()

    while True:
        if sampler.chain.shape[1] > 0:
            sampler.run_mcmc(sampler.chain[:,-1,:], nstep_thin*thin, thin=thin)
        else:
            sampler.run_mcmc(ps, nstep_thin*thin, thin=thin)

        out = bz2.BZ2File(op.join(args.outdir, 'temp_chain.npy.bz2'), 'w')
        try:
            np.save(out, sampler.chain)
        finally:
            out.close()
        os.rename(op.join(args.outdir, 'temp_chain.npy.bz2'),
                  op.join(args.outdir, 'chain.npy.bz2'))

        out = bz2.BZ2File(op.join(args.outdir, 'temp_lnprob.npy.bz2'), 'w')
        try:
            np.save(out, sampler.lnprobability)
        finally:
            out.close()
        os.rename(op.join(args.outdir, 'temp_lnprob.npy.bz2'),
                  op.join(args.outdir, 'lnprob.npy.bz2'))

        tchain = ac.emcee_thinned_chain(sampler.chain)

        if tchain is None:
            print 'After ', sampler.chain.shape[1], ' thinned ensembles, no ACL'
            sys.__stdout__.flush()
            continue
        elif tchain.shape[0]*tchain.shape[1] > args.neff:
            print 'After ', sampler.chain.shape[1], ' thinned ensembles, ', tchain.shape[1], ' independent ensembles'
            print 'Stopping with ', tchain.shape[1]*tchain.shape[0], ' independent samples'
            sys.__stdout__.flush()
            break
        else:
            print 'After ', sampler.chain.shape[1], ' thinned ensembles, ', tchain.shape[1], ' independent ensembles'
            sys.__stdout__.flush()
            continue

        if sampler.chain.shape[0]*sampler.chain.shape[1] > 10*args.neff:
            sampler._chain = sampler._chain[:,1::2,:]
            sampler._lnprob = sampler._lnprob[:,1::2]
            thin *= 2

            out = open(op.join(args.outdir, 'thin.txt'), 'w')
            try:
                out.write('{0:d}\n'.format(thin))
            finally:
                out.close()

            print 'Chain has accumulated too many samples; thinning now by ', thin
            sys.__stdout__.flush()
