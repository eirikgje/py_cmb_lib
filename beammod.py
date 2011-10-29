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
    beam = almmod.ClData(lmax, cls=gout, spectra=spectra, spec_axis=1)
    return beam

class BeamData(object):
    def __init__(self, lmax, beam=None, pol=False, beam_axis=None):
        if beam is not None and beam_axis is not None:
            if beam.shape[beam_axis] != lmax + 1:
                raise ValueError("Explicit beam_axis does not contain the "
                                 "right number of elements")
        if beam_axis is None:
            beam_axis = 0
        self.beam_axis = beam_axis
        self.pol = pol
        if beam is None:
            beam = np.zeros(lmax + 1)
        self._lmax = None
        self.lmax = lmax
        self.beam = beam

    def getbeam(self):
        return self._beam

    def setbeam(self, beam):
        if not isinstance(beam, np.ndarray):
            raise TypeError("Beam must be numpy array")
        if beam.ndim > 2:
            raise ValueError("Beam must have 2 or less dimensions")
        if (self.beam_axis >= beam.ndim or 
                beam.shape[self.beam_axis] != self.lmax + 1):
            #Try to autodetect beam axis
            if beam.shape[0] == self.lmax + 1:
                self.beam_axis = 0
            else:
                if beam.ndim == 2:
                    if beam.shape[1] == self.lmax + 1:
                        self.beam_axis = 1
                    else:
                        raise ValueError("No dimension with appropriate " 
                                         "number of elements")
                else:
                    raise ValueError("No dimension with appropriate number of "
                                     "elements")

        self._beam = beam

    beam = property(getbeam, setbeam)

    def getlmax(self):
        return self._lmax

    def setlmax(self, lmax):
        if self._lmax is not None:
            raise ValueError("Lmax is immutable")
        if not isinstance(lmax, int):
            raise TypeError("lmax must be an integer")
        self._lmax = lmax

    lmax = property(getlmax, setlmax)
