import numpy as np

def good_guess_params(data, bounds, nparams):
    inds = np.searchsorted(bounds, data)-1

    counts = np.bincount(inds, minlength=bounds.shape[0]-1)

    p0s = (counts + 0.5)/(np.sum(counts) + (bounds.shape[0]-1)*0.5)

    params = np.array([np.random.lognormal(mean=np.log(p0s), sigma=1.0/np.sqrt(0.5+counts)) for i in range(nparams)])

    params /= np.sum(params, axis=1).reshape((nparams, 1))

    return params[:,:-1]
        
class HistogramPosterior(object):
    def __init__(self, data, bounds):
        assert np.all(data > bounds[0]), 'data below min bound'
        assert np.all(data < bounds[-1]), 'data above max bound'

        assert np.all(bounds[:-1] < bounds[1:]), 'bounds not sorted'

        self.data = data
        self.bounds = bounds
        
        inds = np.searchsorted(bounds, data)-1

        self.counts = np.bincount(inds, minlength=bounds.shape[0]-1)

    @property
    def nparams(self):
        return self.bounds.shape[0]-2

    def rhos(self, p):
        ps = np.concatenate((p, [1.0-np.sum(p)]))

        return ps / np.diff(self.bounds)

    def density(self, p, xs):
        xs = np.atleast_1d(xs)
        rhos = self.rhos(p)

        inds = np.searchsorted(self.bounds, xs)-1

        outside = (inds<0) | (inds==self.bounds.shape[0])
        inside = ~outside

        ds = np.zeros(xs.shape)

        ds[inside] = rhos[inds[inside]]

        return ds

    def log_prior(self, p):
        if np.any(p < 0):
            return np.NINF

        if np.sum(p) > 1.0:
            return np.NINF

        return -0.5*(np.sum(np.log(p)) + np.log1p(-np.sum(p)))

    def log_likelihood(self, p):
        return np.sum(self.counts*np.log(self.rhos(p)))
        
    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return np.NINF
        else:
            return lp + self.log_likelihood(p)

    def best_guess_params(self):
        return ((self.counts + 0.5)/(np.sum(self.counts) + 0.5*self.counts.shape[0]))[:-1]
