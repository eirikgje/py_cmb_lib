from __future__ import division
import numpy as np

def ind2lm():
    pass

def lm2ind():
    pass

class AlmData(object):
    def __init__(self, lmax, mmax=None, alms=None, lsubd=None, rsubd=None,
                 pol=False):
        if mmax != None:
            raise NotImplementedError()
        self.nnind = lmax * (lmax + 1) / 2 + lmax + 1
        self.dyn_ind = 0
        self._alms = None
        self.subd = ()
        self.pol = pol
        if lsubd is not None:
            self.subdivide(lsubd)
        if rsubd is not None:
            self.subdivide(rsubd, left_of_dyn_d=False)
        if alms is None:
            alms = np.zeros(self.subd[0:self.dyn_ind] + (1,) + 
                           self.subd[self.dyn_ind:] + (self.nnind,) + (2,))
        self.alms = alms
        self.lmax = lmax
        self.mmax = lmax

    def getalms(self):
        return self._alms

    def setalms(self, alms):
        self._alms = self.conform_alms(alms)

    alms = property(getalms, setalms)

    def getlmax(self):
        return self._lmax

    def setlmax(self, lmax):
        if not isinstance(lmax, int):
            raise TypeError("lmax must be an integer")
        else:
            if np.size(self.alms, -2) != lmax * (lmax + 1) / 2 + lmax + 1:
                raise ValueError("""lmax must be compatible with last
                                    dimension of alm""")
            self._lmax = lmax

    lmax = property(getlmax, setlmax)

    def getpol(self):
        return self._pol

    def setpol(self, pol):
        if not isinstance(pol, bool):
            raise TypeError("pol must be a boolean variable")
        if pol:
            if len(self.subd) > 0:
                if not self.subd[-1] == 3:
                    self.subdivide(3, left_of_dyn_d=False)
            else:
                self.subdivide(3, left_of_dyn_d=False)
        self._pol = pol

    pol = property(getpol, setpol)

    def subdivide(self, vals, left_of_dyn_d=True):
        """Can take either int, tuple or numpy arrays as arguments."""
        
#        By default, subdividing will be done in such a way that the dynamical
#        index (i.e. number of alm samples or whatever) is the next-to-last one
#        (the last one being the number of els). Note that adding samples
#        should have nothing to do with subdividing - subdividing is for those
#        cases where each sample will have more than one alm (polarization, 
#        various frequencies etc.) Note, however, that for polarization it is
#        possible to just set the 'pol' keyword to True for the object, and the 
#        subdivision will be taken care of. Also note that after subdividing, 
#        each map array added (or assigned) to the MapData instance must have the
#        shape (x1, x2, ..., xn, n, npix) or (x1, x2, ..., xn, npix) where
#        x1, x2, ..., xn are the subdivisions and n is the number of map samples
#        added (could be 1 and will be interpreted as 1 if missing). 
#        Subdivision can be done several times. The new dimension will then
#        be the leftmost dimension in the resulting MapData.map array.
#
#        """
        old_dyn_ind = self.dyn_ind
        if isinstance(vals, int):
            if left_of_dyn_d:
                self.dyn_ind += 1
        elif isinstance(vals, tuple) or isinstance(vals, np.ndarray):
            if left_of_dyn_d:
                self.dyn_ind += len(vals)
        else:
            raise TypeError('Must be int, tuple or np.ndarray')

        if left_of_dyn_d:
            if isinstance(vals, int):
                self.subd = (vals,) + self.subd
            else:
                self.subd = tuple(vals) + self.subd
        else:
            if isinstance(vals, int):
                self.subd = self.subd + (vals,)
            else:
                self.subd = self.subd + tuple(vals)

        if self.alms is not None:
            self._alms = np.resize(self.alms, self.subd[0:self.dyn_ind] +
                                  (self.alms.shape[old_dyn_ind],) +
                                  self.subd[self.dyn_ind:] +
                                  (self.alms.shape[-2], self.alms.shape[-1]))

    def appendalms(self, alms):
        """Add one or several alms to object instance.

        The input alms(s) must be numpy arrays, or AlmData objects
        and they must have the shape
        (subd, nalms, nels, 2) or (subd, nels, 2) where subd is the current
        subdivision of the AlmData instance, nels is the number
        of elements for l, m >= 0 (lmax * (lmax + 1) / 2 + lmax + 1) for the
        alms already added to the object instance. nalms can be any number, 
        and if this dimension is missing from the array, it will be interpreted 
        as alms for a single map. 
        If there are no subdivisions, a (nels, 2) numpy array is acceptable.

        """
        if isinstance(alms, AlmData):
            if alms.lmax != self.lmax:
                raise ValueError("Lmax is not compatible")
            alms = alms.alms
        if np.size(alms, -1) != 2:
            raise ValueError("Last dimension of alms must be 2")
        if np.size(alms, -2) != np.size(self.alms, -2):
            raise ValueError("Incorrect number of elements in input alms")
        self.alms = np.append(self.alms, self.conform_alms(alms), 
                             axis=self.dyn_ind)

    def conform_alms(self, alms):
        """Make input alms acceptable shape, or raise exception.
        
        Input alms is only compared to the current subdivision, not any present
        alms.
        
        """
        if not isinstance(alms, np.ndarray):
            raise TypeError('Alms must be numpy array')
        mlen = len(self.subd) + 3
        if mlen > alms.ndim + 1:
            raise ValueError('Too few dimensions in alms')
        elif mlen < alms.ndim:
            raise ValueError('Too many dimensions in alms')
        if mlen == alms.ndim:
            #Explicit dynamic dimension
            almsubd = (alms.shape[0:self.dyn_ind] + 
                        alms.shape[self.dyn_ind + 1:-2])
            if (almsubd != self.subd):
                raise ValueError("""Alm dimensions do not conform to AlmData
                                subdivision""")
        elif mlen == alms.ndim + 1:
            #Dynamic dimension is implicit
            almsubd = (alms.shape[0:-2])
            if (almsubd  != self.subd):
                raise ValueError("""Alm dimensions do not conform to AlmData
                                subdivision""")
            else:
                alms = alms.reshape(self.subd[0:self.dyn_ind] + (1,) + 
                                  self.subd[self.dyn_ind:] + 
                                  alms.shape[-2:])
        return alms
