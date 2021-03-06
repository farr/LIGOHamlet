import foreground as fg
import numpy as np
import scipy.optimize as so
import scipy.special as sp
import warnings as warn

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

        # alpha ~ effective number of "counts" distributed uniformly
        # by the prior
        alpha_counts = 1.0
        alphax_counts = 1.0
        alphay_counts = 1.0

        alphas = np.outer(dx, dy)
        alphas *= alpha_counts/np.sum(alphas)

        self.alphas = alphas[self.uinds[:,0], self.uinds[:,1]]
        self.alphas_rem = alpha_counts - np.sum(self.alphas)

        bgx_alphas = dx.copy()
        bgx_alphas *= alphax_counts/np.sum(bgx_alphas)

        bgy_alphas = dy.copy()
        bgy_alphas *= alphay_counts/np.sum(bgy_alphas)

        self.bgx_alphas = bgx_alphas[self.uxinds]
        self.bgx_alphas_rem = alphax_counts - np.sum(self.bgx_alphas)

        self.bgy_alphas = bgy_alphas[self.uyinds]
        self.bgy_alphas_rem = alphay_counts - np.sum(self.bgy_alphas)
        

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

    def log_rhof(self, p):
        p = self.to_params(p)

        log_rhofs = np.zeros((self.foreground.xbins.shape[0]-1, self.foreground.ybins.shape[0]-1))
        log_rhofs[:,:] = np.NINF
        log_rhofs[self.uinds[:,0], self.uinds[:,1]] = p['log_fg_ps'] # Only fill in the relevant bins
        return log_rhofs

    def log_rhob(self, p):
        p = self.to_params(p)

        log_rhobs = np.zeros((self.foreground.xbins.shape[0]-1, self.foreground.ybins.shape[0]-1))
        log_rhobs[:,:] = np.NINF
        log_rhobs[self.uxinds, :] = p['log_bgx_ps'].reshape((-1, 1))
        log_rhobs[:,self.uyinds] += p['log_bgy_ps'].reshape((1, -1))

        return log_rhobs

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
        
        log_rhofs = self.log_rhof(p)
        log_rhobs = self.log_rhob(p)

        ll += np.sum(np.logaddexp(np.log(p['Rf']) + log_rhofs[self.xinds, self.yinds],
                                  np.log(p['Rb']) + log_rhobs[self.xinds, self.yinds]))

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

        # Correct for zeros in counts
        sel = self.counts == 0
        if np.any(sel):
            ps['log_fg_ps'][sel] = sp.psi(self.alphas[sel])

        sel = self.bgxcounts == 0
        if np.any(sel):
            ps['log_bgx_ps'][sel] = sp.psi(self.bgx_alphas[sel])

        sel = self.bgycounts == 0
        if np.any(sel):
            ps['log_bgy_ps'][sel] = sp.psi(self.bgy_alphas[sel])

        log_rhof = self.log_rhof(ps)[self.xinds, self.yinds]
        log_rhob = self.log_rhob(ps)[self.xinds, self.yinds]

        log_Rf = np.logaddexp.reduce(log_rhof - np.logaddexp(log_rhof, log_rhob))
        log_Rb = np.logaddexp.reduce(log_rhob - np.logaddexp(log_rhof, log_rhob))

        ps['Rf'] = np.exp(log_Rf)
        ps['Rb'] = np.exp(log_Rb)

        return ps.reshape((1,)).view(float).reshape((-1,))

    def log_pfores(self, p):
        p = self.to_params(p)

        log_rhofs = self.log_rhof(p)[self.xinds, self.yinds]
        log_rhobs = self.log_rhob(p)[self.xinds, self.yinds]

        return np.log(p['Rf']) + log_rhofs - np.logaddexp(np.log(p['Rf']) + log_rhofs,
                                                          np.log(p['Rb']) + log_rhobs)

    def pfores(self, p):
        return np.exp(self.log_pfores(p))

    def log_pbacks(self, p):
        p = self.to_params(p)

        log_rhofs = self.log_rhof(p)[self.xinds, self.yinds]
        log_rhobs = self.log_rhob(p)[self.xinds, self.yinds]

        return np.log(p['Rb']) + log_rhobs - np.logaddexp(np.log(p['Rf']) + log_rhofs,
                                                          np.log(p['Rb']) + log_rhobs)

    def pbacks(self, p):
        return np.exp(self.log_pbacks(p))

class AccuracyWarning(Warning):
    pass

class SimplifiedPosterior(Posterior):
    def __init__(self, *args, **kwargs):
        super(SimplifiedPosterior, self).__init__(*args, **kwargs)

        log_fgdenom = np.log(np.sum(self.counts) + self.nfg_rem + np.sum(self.alphas) + self.alphas_rem)
        log_bgxdenom = np.log(np.sum(self.bgxcounts) + self.nbgx_rem + np.sum(self.bgx_alphas) + self.bgx_alphas_rem)
        log_bgydenom = np.log(np.sum(self.bgycounts) + self.nbgy_rem + np.sum(self.bgy_alphas) + self.bgy_alphas_rem)

        self.log_fg_ps = np.log(self.alphas + self.counts) - log_fgdenom
        self.log_bgx_ps = np.log(self.bgx_alphas + self.bgxcounts) - log_bgxdenom
        self.log_bgy_ps = np.log(self.bgy_alphas + self.bgycounts) - log_bgydenom

        p0 = self.params_guess()
        self.log_coinc_fg_ps = self.log_rhof(p0)[self.xinds, self.yinds]
        self.log_coinc_bg_ps = self.log_rhob(p0)[self.xinds, self.yinds]
        

        if np.any(self.counts < 10):
            warn.warn(AccuracyWarning('foreground may be inaccurately estimated---use Posterior'))
        if np.any(self.bgxcounts < 10):
            warn.warn(AccuracyWarning('x-background may be inaccurately estimated---use Posterior'))

        if np.any(self.bgycounts < 10):
            warn.warn(AccuracyWarning('y-background may be inaccurately estimated---use Posterior'))

        if self.uxinds.shape[0] < self.xinds.shape[0]:
            warn.warn(AccuracyWarning('more than one coinc per x-bin'))
        if self.uyinds.shape[0] < self.yinds.shape[0]:
            warn.warn(AccuracyWarning('more than one coinc per y-bin'))

    @property
    def nparams(self):
        return 2

    def to_params(self, p):
        pp = np.zeros(super(SimplifiedPosterior, self).nparams)
	p = np.atleast_1d(p)
	p = p.view(float)

        pp[0] = p[0]
	pp[1] = p[1]

        pp = super(SimplifiedPosterior, self).to_params(pp)

        pp['log_fg_ps'] = self.log_fg_ps
        pp['log_bgx_ps'] = self.log_bgx_ps
        pp['log_bgy_ps'] = self.log_bgy_ps

        return pp

    def params_guess(self):
        p = super(SimplifiedPosterior, self).params_guess()

        return p[:2]

    def log_prior(self, p):
        if p[0] < 0:
            return np.NINF
        if p[1] < 0:
            return np.NINF

        return -0.5*(np.log(p[0]) + np.log(p[1]))

    def log_likelihood(self, p):
        log_coinc_ps = np.logaddexp(np.log(p[0]) + self.log_coinc_fg_ps,
                                    np.log(p[1]) + self.log_coinc_bg_ps)
        
        return np.sum(log_coinc_ps) - p[0] - p[1]
