from __future__ import division
import numpy as np
import almmod

def gaussbeam(fwhm, lmax, ndim=1):
    """ fwhm in arcmin """

    sigma = fwhm/(60.0 * 180.0) * np.pi / np.sqrt(8.0*np.log(2.0))
    l = np.arange(lmax + 1)
    sig2 = sigma ** 2
    g = np.exp(-0.5*l*(l+1.0)*sig2)
    g = np.resize(g, (ndim, g.size))
    g = g.swapaxes(0, 1)
    if ndim > 1:
        if ndim > 4:
            raise ValueError("ndim must be 4 or less")
    if ndim == 4:
        factor_pol = np.exp([0.0, 2.0*sig2, 2.0*sig2, sig2])
        gout = g * factor_pol[0:ndim]
        spectra = ['TT', 'EE', 'BB', 'TE']
    elif ndim == 1:
        gout = g
        spectra = 'temp'
    else:
        raise NotImplementedError()
    beam = almmod.ClData(lmax, cls=gout, spectra=spectra)
    return beam
