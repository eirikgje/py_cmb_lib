from __future__ import division
import numpy as np
import almmod

def gaussbeam(fwhm, lmax, pol=False):
    """ fwhm in arcmin """
    if pol:
        ndim = 3
    else:
        ndim = 1

    sigma = fwhm/(60.0 * 180.0) * np.pi / np.sqrt(8.0*np.log(2.0))
    l = np.arange(lmax + 1)
    sig2 = sigma ** 2
    g = np.exp(-0.5*l*(l+1.0)*sig2)
    g = np.resize(g, (ndim, g.size))
    g = g.swapaxes(0, 1)
    if ndim == 3:
        factor_pol = np.exp([0.0, 2.0*sig2, 2.0*sig2])
        gout = g * factor_pol
        pol_axis = 1
    elif ndim == 1:
        gout = g
        pol_axis = None
    beam = BeamData(lmax, beam=gout, pol_axis=pol_axis, beam_axis=0)
    return beam

class BeamData(object):
    def __init__(self, lmax, beam=None, pol_axis=None, beam_axis=None):
        if beam is not None and beam_axis is not None:
            if beam.shape[beam_axis] != lmax + 1:
                raise ValueError("Explicit beam_axis does not contain the "
                                 "right number of elements")
        if beam_axis is None:
            beam_axis = 0
        self.beam_axis = beam_axis
        if beam is None:
            beam = np.zeros(lmax + 1)
        self._lmax = None
        self.lmax = lmax
        self.beam = beam
        self.pol_axis = pol_axis

    def __getitem__(self, index):
        return self.beam[index]

    def __setitem__(self, key, item):
        self.beam[key] = item

    def getpol_axis(self):
        if self._pol_axis is not None:
            if self.beam.shape[self._pol_axis] != 3:
                raise ValueError("Polarization axis has not been updated since"
                                 "changing number of beam dimensions")
        return self._pol_axis

    def setpol_axis(self, pol_axis):
        if pol_axis is not None:
            if self.beam.shape[pol_axis] != 3:
                self._pol_axis = None
                raise ValueError("Polarization axis does not have 3 dimensions")
        self._pol_axis = pol_axis

    pol_axis = property(getpol_axis, setpol_axis)

    def getlmax(self):
        return self._lmax

    def setlmax(self, lmax):
        if self._lmax is not None:
            raise ValueError("Lmax is immutable")
        if not isinstance(lmax, int):
            raise TypeError("lmax must be an integer")
        self._lmax = lmax

    lmax = property(getlmax, setlmax)

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
