from __future__ import division
import numpy as np

def ind2lm:
    pass

def lm2ind:
    pass

class AlmData:
    def __init__(self, lmax, mmax=lmax, alms=None, lsubd=None, rsubd=None,
                 pol=False):
        if mmax != lmax:
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
