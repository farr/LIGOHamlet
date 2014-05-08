import foreground as fg
import numpy as np
import scipy.optimize as so

class Posterior(object):
    def __init__(self, coincs, bgs, snr_min, N=10000000):
        self.snr_min = snr_min
        self.coincs = coincs
        self.bgs = bgs
        self.foreground = fg.Foreground(snr_min, N)

        self.xinds = np.searchsorted(self.foreground.xbins, coincs[:,0])-1
        self.yinds = np.searchsorted(self.foreground.ybins, coincs[:,1])-1

        self.uxinds = np.unique(self.xinds)
        self.uyinds = np.unique(self.yinds)

        ind_set = set([xy for xy in zip(self.xinds, self.yinds)])
        self.uinds = np.array([xy for xy in ind_set])

        # The model assumes that at least one bin in each distribution
        # does not have a coinc in it!
        if self.uxinds.shape[0] >= self.foreground.xbins.shape[0]-1:
            raise ValueError('too many unique x indices')
        if self.uyinds.shape[0] >= self.foreground.ybins.shape[0]-1:
            raise ValueError('too many unique y indices')
        if self.uinds.shape[0] >= (self.foreground.xbins.shape[0]-1)*(self.foreground.ybins.shape[0]-1):
            raise ValueError('too many xy indices')

        self.all_bgxcounts, _ = np.histogram(bgs[0], bins=self.foreground.xbins)
        self.all_bgycounts, _ = np.histogram(bgs[1], bins=self.foreground.ybins)

        self.bgxcounts = self.all_bgxcounts[self.uxinds]
        self.bgycounts = self.all_bgycounts[self.uyinds]
        self.counts = self.foreground.counts[self.uinds[:,0], self.uinds[:,1]]

        self.nfg_rem = np.sum(self.foreground.counts) - np.sum(self.counts)
        self.nbgx_rem = bgs[0].shape[0] - np.sum(self.bgxcounts)
        self.nbgy_rem = bgs[1].shape[0] - np.sum(self.bgycounts)

    @property
    def nparams(self):
        return 2 + self.uxinds.shape[0] + self.uyinds.shape[0] + self.uinds.shape[0]

    @property
    def dtype(self):
        return np.dtype([('Rf', np.float),
                         ('Rb', np.float),
                         ('fg_ps', np.float, self.uinds.shape[0]),
                         ('bgx_ps', np.float, self.uxinds.shape[0]),
                         ('bgy_ps', np.float, self.uyinds.shape[0])])

    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def log_prior(self, p):
        p = self.to_params(p)

        if p['Rf'] < 0:
            return np.NINF
        if p['Rb'] < 0:
            return np.NINF

        if np.any(p['fg_ps'] < 0):
            return np.NINF
        if np.any(p['bgx_ps'] < 0):
            return np.NINF
        if np.any(p['bgy_ps'] < 0):
            return np.NINF

        if np.sum(p['fg_ps']) > 1.0:
            return np.NINF
        if np.sum(p['bgx_ps']) > 1.0:
            return np.NINF
        if np.sum(p['bgy_ps']) > 1.0:
            return np.NINF

        return -0.5*(np.sum(np.log(p['fg_ps'])) +
                     np.sum(np.log(p['bgx_ps'])) +
                     np.sum(np.log(p['bgy_ps'])) +
                     np.log1p(-np.sum(p['fg_ps'])) +
                     np.log1p(-np.sum(p['bgx_ps'])) +
                     np.log1p(-np.sum(p['bgy_ps'])) +
                     np.log(p['Rf']) +
                     np.log(p['Rb']))

    def log_likelihood(self, p):
        p = self.to_params(p)

        ll = 0.0

        ll += np.sum(self.bgxcounts*np.log(p['bgx_ps'])) + self.nbgx_rem*np.log1p(-np.sum(p['bgx_ps']))

        ll += np.sum(self.bgycounts*np.log(p['bgy_ps'])) + self.nbgy_rem*np.log1p(-np.sum(p['bgy_ps']))

        ll += np.sum(self.counts*np.log(p['fg_ps'])) + self.nfg_rem*np.log1p(-np.sum(p['fg_ps']))
        
        rhofs = np.zeros((self.foreground.xbins.shape[0]-1, self.foreground.ybins.shape[0]-1))
        rhobs = rhofs.copy()

        rhofs[self.uinds[:,0], self.uinds[:,1]] = p['fg_ps']

        rhobs[self.uxinds, :] = p['bgx_ps'].reshape((-1, 1))
        rhobs[:,self.uyinds] *= p['bgy_ps'].reshape((1, -1))

        ll += np.sum(np.logaddexp(np.log(p['Rf']) + np.log(rhofs[self.xinds, self.yinds]),
                                  np.log(p['Rb']) + np.log(rhobs[self.xinds, self.yinds])))

        ll -= p['Rf']
        ll -= p['Rb']

        return ll

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return np.NINF
        else:
            return lp + self.log_likelihood(p)

    def params_guess(self):
        nf = np.sum(self.foreground.counts + 0.5)
        nbx = np.sum(self.all_bgxcounts + 0.5)
        nby = np.sum(self.all_bgycounts + 0.5)

        ps = self.to_params(np.zeros(self.nparams))

        ps['fg_ps'] = (self.counts + 0.5)/nf
        ps['bgx_ps'] = (self.bgxcounts + 0.5)/nbx
        ps['bgy_ps'] = (self.bgycounts + 0.5)/nby

        return ps.reshape((1,)).view(float).reshape((-1,))
