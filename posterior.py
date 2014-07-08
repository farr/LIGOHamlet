import foreground as fg
import numpy as np
import scipy.optimize as so

class Posterior(object):
    def __init__(self, coincs, bgs, snr_min, N=10000000, foreground=None):
        self.snr_min = snr_min
        self.coincs = coincs
        self.bgs = bgs

        if foreground is None:
            self.foreground = fg.Foreground(snr_min, N)
        else:
            self.foreground = foreground

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

        dx = np.diff(self.foreground.xbins)
        dy = np.diff(self.foreground.ybins)

        alphas = np.outer(dx, dy)
        alphas *= 0.5*dx.shape[0]*dy.shape[0]/np.sum(alphas)

        self.alphas = alphas[self.uinds[:,0], self.uinds[:,1]]
        self.alphas_rem = 0.5*dx.shape[0]*dy.shape[0] - np.sum(self.alphas)

        bgx_alphas = dx.copy()
        bgx_alphas *= 0.5*dx.shape[0]/np.sum(bgx_alphas)

        bgy_alphas = dy.copy()
        bgy_alphas *= 0.5*dy.shape[0]/np.sum(bgy_alphas)

        self.bgx_alphas = bgx_alphas[self.uxinds]
        self.bgx_alphas_rem = 0.5*dx.shape[0] - np.sum(self.bgx_alphas)

        self.bgy_alphas = bgy_alphas[self.uyinds]
        self.bgy_alphas_rem = 0.5*dy.shape[0] - np.sum(self.bgy_alphas)
        

    @property
    def nparams(self):
        return 2 + self.uxinds.shape[0] + self.uyinds.shape[0] + self.uinds.shape[0]

    @property
    def dtype(self):
        return np.dtype([('Rf', np.float),
                         ('Rb', np.float),
                         ('log_fg_ps', np.float, self.uinds.shape[0]),
                         ('log_bgx_ps', np.float, self.uxinds.shape[0]),
                         ('log_bgy_ps', np.float, self.uyinds.shape[0])])

    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def rhof(self, p):
        p = self.to_params(p)

        rhofs = np.zeros((self.foreground.xbins.shape[0]-1, self.foreground.ybins.shape[0]-1))
        rhofs[self.uinds[:,0], self.uinds[:,1]] = np.exp(p['log_fg_ps']) # Only fill in the relevant bins
        
        return rhofs

    def rhob(self, p):
        p = self.to_params(p)

        rhobs = np.zeros((self.foreground.xbins.shape[0]-1, self.foreground.ybins.shape[0]-1))
        rhobs[self.uxinds, :] = np.exp(p['log_bgx_ps'].reshape((-1, 1)))
        rhobs[:,self.uyinds] *= np.exp(p['log_bgy_ps'].reshape((1, -1)))

        return rhobs

    def log_prior(self, p):
        p = self.to_params(p)

        if p['Rf'] < 0:
            return np.NINF
        if p['Rb'] < 0:
            return np.NINF

        if np.logaddexp.reduce(p['log_fg_ps']) > 0.0:
            return np.NINF
        if np.logaddexp.reduce(p['log_bgx_ps']) > 0.0:
            return np.NINF
        if np.logaddexp.reduce(p['log_bgy_ps']) > 0.0:
            return np.NINF

        return np.sum((self.alphas - 1)*p['log_fg_ps']) + \
            np.sum((self.bgx_alphas - 1)*p['log_bgx_ps']) + \
            np.sum((self.bgy_alphas - 1)*p['log_bgy_ps']) + \
            (self.alphas_rem-1)*np.log1p(-np.exp(np.logaddexp.reduce(p['log_fg_ps']))) + \
            (self.bgx_alphas_rem-1)*np.log1p(-np.exp(np.logaddexp.reduce(p['log_bgx_ps']))) + \
            (self.bgy_alphas_rem-1)*np.log1p(-np.exp(np.logaddexp.reduce(p['log_bgy_ps']))) + \
            -0.5*(np.log(p['Rf']) + np.log(p['Rb'])) + \
            np.sum(p['log_fg_ps']) + np.sum(p['log_bgx_ps']) + np.sum(p['log_bgy_ps']) # Jacobian for logs.

    def log_likelihood(self, p):
        p = self.to_params(p)

        ll = 0.0

        ll += np.sum(self.bgxcounts*p['log_bgx_ps']) + self.nbgx_rem*np.log1p(-np.exp(np.logaddexp.reduce(p['log_bgx_ps'])))

        ll += np.sum(self.bgycounts*p['log_bgy_ps']) + self.nbgy_rem*np.log1p(-np.exp(np.logaddexp.reduce(p['log_bgy_ps'])))

        ll += np.sum(self.counts*p['log_fg_ps']) + self.nfg_rem*np.log1p(-np.exp(np.logaddexp.reduce(p['log_fg_ps'])))
        
        rhofs = self.rhof(p)
        rhobs = self.rhob(p)

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
        nf = np.sum(self.foreground.counts) + np.sum(self.alphas) + self.alphas_rem
        nbx = np.sum(self.all_bgxcounts) + np.sum(self.bgx_alphas) + self.bgx_alphas_rem
        nby = np.sum(self.all_bgycounts) + np.sum(self.bgy_alphas) + self.bgy_alphas_rem

        ps = self.to_params(np.zeros(self.nparams))

        ps['log_fg_ps'] = np.log((self.counts + self.alphas)/nf)
        ps['log_bgx_ps'] = np.log((self.bgxcounts + self.bgx_alphas)/nbx)
        ps['log_bgy_ps'] = np.log((self.bgycounts + self.bgy_alphas)/nby)

        return ps.reshape((1,)).view(float).reshape((-1,))

    def pfores(self, p):
        p = self.to_params(p)

        rhofs = self.rhof(p)
        rhobs = self.rhob(p)

        return p['Rf']*rhofs[self.xinds, self.yinds]/(p['Rf']*rhofs[self.xinds, self.yinds] + p['Rb']*rhobs[self.xinds, self.yinds])

    def pbacks(self, p):
        p = self.to_params(p)

        rhofs = self.rhof(p)
        rhobs = self.rhob(p)

        return p['Rb']*rhobs[self.xinds, self.yinds]/(p['Rf']*rhofs[self.xinds, self.yinds] + p['Rb']*rhobs[self.xinds, self.yinds])
