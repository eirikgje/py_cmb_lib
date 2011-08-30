from __future__ import division
import numpy as np

def gaussbeam(fwhm, lmax, dim=1):
    """ fwhm in arcmin """

    sigma = fwhm/(60.0 * 180.0) * np.pi / np.sqrt(8.0*np.log(2.0))
    l = np.arange(lmax + 1)
    sig2 = sigma ** 2
    g = np.exp(-0.5*l*(l+1.0)*sig2)
    g = np.resize(g, (dim, g.size))
    g = g.swapaxes(0, 1)
    if dim > 1:
        if dim > 4:
            raise ValueError("dim must be 4 or less")
        factor_pol = np.exp([0.0, 2.0*sig2, 2.0*sig2, sig2])
        gout = g * factor_pol[0:dim]
    elif dim == 1 or dim == 0:
        gout = g
    else:
        raise ValueError("dim must be 0 or greater")
    return gout
