import lal
import numpy as np

def draw_normalised_snrs():
    r"""Returns ``(rho_H, rho_L)``, with :math:`0 \leq \rho \leq 1` for the
    SNR in Hanford and Livingston drawn fairly over sky positions,
    time, polarisation, and inclination.

    """

    LLO_index = lal.LALDetectorIndexLLODIFF
    LHO_index = lal.LALDetectorIndexLHODIFF

    gmst = np.random.uniform(low=0.0, high=2.0*np.pi)
    ra = np.random.uniform(low=0.0, high=2.0*np.pi)
    dec = np.arcsin(np.random.uniform(low=-1.0, high=1.0))
    psi = np.random.uniform(low=0.0, high=2.0*np.pi)
    cos_i = np.random.uniform(low=-1.0, high=1.0)

    pfac = 0.5*(1.0+cos_i*cos_i)
    cfac = cos_i

    pfac2 = pfac*pfac
    cfac2 = cfac*cfac
    
    fpl, fcl = lal.ComputeDetAMResponse(lal.CachedDetectors[LLO_index].response,
                                        ra, dec, psi, gmst)
    fph, fch = lal.ComputeDetAMResponse(lal.CachedDetectors[LHO_index].response,
                                        ra, dec, psi, gmst)

    return np.sqrt(fph*fph*pfac2 + fch*fch*cfac2), np.sqrt(fpl*fpl*pfac2 + fcl*fcl*cfac2)

def draw_snrs(snr_thresh, nsnr):
    """Returns an array of shape ``(N, 2)`` giving the SNR in H and L for
    signals where both detectors are above ``snr_thresh``.

    """
    us = np.random.uniform(size=nsnr)

    snr0s = snr_thresh/(1.0-us)**(1.0/3.0) 
    snrs = np.array([draw_normalised_snrs() for i in range(nsnr)])*snr0s.reshape((-1, 1))

    return snrs[(snrs[:,0] > snr_thresh) & (snrs[:,1] > snr_thresh), :]

def dsnr(snr, snr_thresh, N):
    return 0.2886751347*snr**2.5*N**-0.25/snr_thresh**1.5

def snr_bins(snr_max, snr_thresh, N):
    bins = [snr_thresh]

    while bins[-1] < snr_max:
        bins.append(bins[-1]+dsnr(bins[-1], snr_thresh, N))

    return np.array(bins)

def snr_histogram(snr_thresh, N):
    if N < 100000:
        raise ValueError('you won\'t get far with such a small N')
    
    Ns = np.round(np.linspace(0, N, N/100000)).astype(np.int)
    Ndraws = np.diff(Ns)

    xbins = None
    ybins = None
    counts = None

    for n in Ndraws:
        snrs = draw_snrs(snr_thresh, n)

        xmax = np.max(snrs[:,0])
        ymax = np.max(snrs[:,1])

        xbinsn = snr_bins(xmax, snr_thresh, N)
        ybinsn = snr_bins(ymax, snr_thresh, N)

        countsn, xbinsn, ybinsn = np.histogram2d(snrs[:,0], snrs[:,1], bins=[xbinsn, ybinsn])

        xbins, ybins, counts = combine_histograms(xbins, ybins, counts,
                                                  xbinsn, ybinsn, countsn)

    return xbins, ybins, counts

def combine_histograms(xbins1, ybins1, counts1, xbins2, ybins2, counts2):
    if xbins1 is None or ybins1 is None or counts1 is None:
        return xbins2, ybins2, counts2
    elif xbins2 is None or ybins2 is None or counts2 is None:
        return xbins1, ybins1, counts1
    else:
        imax = max(counts1.shape[0], counts2.shape[0])
        jmax = max(counts1.shape[1], counts2.shape[1])

        counts = np.zeros((imax, jmax), dtype=np.int)
        counts[:counts1.shape[0], :counts1.shape[1]] += counts1
        counts[:counts2.shape[0], :counts2.shape[1]] += counts2

        if xbins1.shape[0] > xbins2.shape[0]:
            xbins = xbins1
        else:
            xbins = xbins2

        if ybins1.shape[0] > ybins2.shape[0]:
            ybins = ybins1
        else:
            ybins = ybins2

        return xbins, ybins, counts

class Foreground(object):
    def __init__(self, snr_thresh, N):
        xbins, ybins, counts = snr_histogram(snr_thresh, N)

        self.snr_thresh = snr_thresh

        self.xbins = xbins
        self.ybins = ybins
        self.counts = counts

        dxs = np.diff(self.xbins)
        dys = np.diff(self.ybins)

        areas = np.outer(dxs, dys)
        alphas = areas.copy()
        alphas *= 0.5*alphas.shape[0]*alphas.shape[1]/np.sum(alphas)

        self.mean_density = (counts+alphas)/np.sum(counts + alphas)/dxs.reshape((-1, 1))/dys.reshape((1, -1))

    def get_counts(self, pts):
        pts = np.atleast_2d(pts)

        ii = np.searchsorted(self.xbins, pts[:,0])-1
        jj = np.searchsorted(self.ybins, pts[:,1])-1

        return ii, jj, self.counts[ii, jj]
    
