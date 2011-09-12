from __future__ import division
import numpy as np

def ind2lm(i):
    if not isinstance(i, int):
        raise TypeError("index must be integer")
    l = int((-1 + np.sqrt(1 + 8 * i)) // 2)
    m = int(i - l * (l + 1) // 2)
    return (l, m)
    
def lm2ind(lm):
    if not all([isinstance(j, int) for j in lm]):
        raise TypeError("l, m must be integers")
    return lm[0] * (lm[0] + 1) // 2 + lm[1]

class AlmData(object):
    def __init__(self, lmax, mmax=None, alms=None, ind_axis=None,
                 pol_axis=None, pol_iter=False):
        if alms is not None and ind_axis is not None:
            if alms[ind_axis] != lmax * (lmax + 1) // 2 + lmax + 1:
                raise ValueError("""Explicit ind_axis does not contain right
                                    number of elements""")
        if ind_axis is None:
            ind_axis = 0
        self.ind_axis = ind_axis
        self._lmax = None
        if mmax != lmax and mmax is not None:
            raise NotImplementedError()
        self.nnind = lmax * (lmax + 1) // 2 + lmax + 1
        if alms is None:
            alms = np.zeros(self.nnind, dtype=np.complex)
        self.lmax = lmax
        self.mmax = lmax
        self.alms = alms
        self.pol_axis = pol_axis
        self.pol_iter = pol_iter

    def __iter__(self):
        return _alms_iter(self)

    def getalms(self):
        return self._alms

    def setalms(self, alms):
        if not isinstance(alms, np.ndarray):
            raise TypeError("Alms must be numpy array")
        if not alms.dtype == np.complex:
            raise TypeError("Alms must be complex")
        if (self.ind_axis >= alms.ndim 
                or alms.shape[self.ind_axis] != self.nnind):
            #Try to autodetect pixel axis
            for i in range(alms.ndim):
                if alms.shape[i] == self.nnind:
                    self.ind_axis = i
                    break
            else:
                raise ValueError("""Index number of input alms does not conform 
                                    to lmax""")
        self._alms = alms

    alms = property(getalms, setalms)

    def getlmax(self):
        return self._lmax

    def setlmax(self, lmax):
        if self._lmax is not None:
            raise ValueError("Lmax is immutable")
        if not isinstance(lmax, int):
            raise TypeError("lmax must be an integer")
        self._lmax = lmax

    lmax = property(getlmax, setlmax)

    def getpol_axis(self):
        if self._pol_axis is not None:
            if self.alms.shape[self._pol_axis] != 3:
                raise ValueError("""Polarization axis has not been updated since
                                    changing number of alm dimensions""")
        return self._pol_axis

    def setpol_axis(self, pol_axis):
        if pol_axis is not None:
            if self.alms.shape[pol_axis] != 3:
                self._pol_axis = None
                raise ValueError("Polarization axis does not have 3 dimensions")
        self._pol_axis = pol_axis

    pol_axis = property(getpol_axis, setpol_axis)

    def appendalms(self, alms, along_axis=0):
        """Add one or several alms to object instance.

        The alms must be numpy arrays or AlmData objects, and along_axis
        signifies the axis along which to append the alms. If one of the alms
        has one dimension more than the other, along_axis will be interpreted
        to hold for that alms, and the 'shorter' alms will be reshaped before 
        appending.
        """

        if isinstance(alms, AlmData):
            if alms.lmax != self.lmax:
                raise ValueError("Lmax is not compatible")
            alms = alms.alms

        if alms.ndim == self.alms.ndim:
            pass
        elif alms.ndim == self.alms.ndim + 1:
            self.alms = self.alms.reshape(self.alms.shape[0:along_axis] + (1,) +
                                        self.alms.shape[along_axis:])
        elif alms.ndim == self.alms.ndim - 1:
            alms = alms.reshape(alms.shape[0:along_axis] + (1,) + 
                              alms.shape[along_axis:])
        else:
            raise ValueError("Incompatible number of dimensions between alms")

        if along_axis == self.ind_axis:
            raise ValueError("Cannot append along index axis")
        if self.pol_axis is not None:
            if along_axis == self.pol_axis:
                raise ValueError("Cannot append along polarization axis")

        self.alms = np.append(self.alms, alms, axis=along_axis)

class _alms_iter(object):
    def __init__(self, ad):
        if not isinstance(ad, AlmData):
            raise TypeError()
        self._curralms = 1
        if ad.pol_iter == True and ad.pol_axis is None:
            raise ValueError("pol_iter is True but no pol_axis given")
        self._pol_iter = ad.pol_iter
        if self._pol_iter:
            if ad.ind_axis < ad.pol_axis:
                self._subshape = list(ad.alms.shape[:ad.ind_axis] + 
                                    ad.alms.shape[ad.ind_axis + 1:ad.pol_axis] +
                                    ad.alms.shape[ad.pol_axis + 1:])
            else:
                self._subshape = list(ad.alms.shape[:ad.pol_axis] + 
                                    ad.alms.shape[ad.pol_axis + 1:ad.ind_axis] +
                                    ad.alms.shape[ad.ind_axis + 1:])
        else:
            self._subshape = list(ad.alms.shape[:ad.ind_axis] + 
                            ad.alms.shape[ad.ind_axis + 1:])
        for dim in self._subshape:
            self._curralms *= dim
        self._alms = ad.alms
        self._ind_axis = ad.ind_axis
        self._pol_axis = ad.pol_axis
        if self._pol_iter:
            if self._pol_axis < self._ind_axis:
                self._ind_axis -= 1
            else:
                self._pol_axis -= 1
        #Copies subshape
        self._currind = list(self._subshape)

    def next(self):
        if self._curralms == 0:
            raise StopIteration()
        trace_ind = len(self._subshape) - 1
        if self._currind == self._subshape:
            #First iteration
            self._currind = list(np.zeros(len(self._subshape), dtype=int))
        else:
            while (self._currind[trace_ind] == self._subshape[trace_ind] - 1):
                self._currind[trace_ind] = 0
                trace_ind -= 1
            self._currind[trace_ind] += 1
        self._curralms -= 1
        if self._pol_iter:
            if self._ind_axis < self._pol_axis:
                return self._alms[self._currind[:self._ind_axis] + [Ellipsis,]
                            + self._currind[self._ind_axis:self._pol_axis] 
                            + [Ellipsis,] + self._currind[self._pol_axis:]]
            else:
                return self._alms[self._currind[:self._pol_axis] + [Ellipsis,]
                            + self._currind[self._pol_axis:self._ind_axis] 
                            + [Ellipsis,] + self._currind[self._ind_axis:]]
        else:
            return self._alms[self._currind[:self._ind_axis] + [Ellipsis,] 
                        + self._currind[self._ind_axis:]]

class ClData(object):
    def __init__(self, lmax, cls=None, spectra='temp', spec_axis=None, 
                 cl_axis=None, spec_iter=False):
        if cls is not None and cl_axis is not None:
            if cls.shape[cl_axis] != lmax + 1:
                raise ValueError("""Explicit cl_axis does not contain the right
                                    number of elements""")
        if cl_axis is None:
            cl_axis = 0
        self.cl_axis = cl_axis
        self.spectra = spectra
        if cls is None:
            cls = np.zeros(lmax + 1)
        self._lmax = None
        self.lmax = lmax
        self.cls = cls
        self.spec_axis = spec_axis
        self.spec_iter = spec_iter

    def __iter__(self):
        return _cls_iter(self)

    def getcls(self):
        return self._cls

    def setcls(self, cls):
        if not isinstance(cls, np.ndarray):
            raise TypeError("Cls must be numpy array")
        if self.cl_axis >= cls.ndim or cls.shape[self.cl_axis] != self.lmax + 1:
            #Try to autodetect cl-axis
            for i in range(cls.ndim):
                if cls.shape[i] == self.lmax + 1:
                    self.cl_axis = i
                    break
            else:
                raise ValueError("""Index number of input cls does not conform 
                                    to lmax""")
        self._cls = cls

    cls = property(getcls, setcls)

    def getlmax(self):
        return self._lmax

    def setlmax(self, lmax):
        if self._lmax is not None:
            raise ValueError("Lmax is immutable")
        if not isinstance(lmax, int):
            raise TypeError("lmax must be an integer")
        self._lmax = lmax

    lmax = property(getlmax, setlmax)

    def getspectra(self):
        return self._spectra

    def setspectra(self, spectra):
        if isinstance(spectra, str):
            if spectra.lower() == 'all':
                spectra = ['TT', 'TE', 'TB', 'EE', 'EB', 'BB']
            elif spectra.lower() == 't-e':
                spectra = ['TT', 'TE', 'EE']
            elif spectra.lower() == 'temp':
                spectra = ['TT']
            else:
                raise TypeError("""Setting the spectra manually must be done
                                  using a list, or one of the predefined
                                  keywords""")
        elif isinstance(spectra, list):
            approved = ['TT', 'TE', 'TB', 'EE', 'EB', 'BB']
            for spectrum in spectra:
                if not isinstance(spectrum, str):
                    raise TypeError("""The spectra in the spectrum list must
                                      be strings""")
                if not spectrum in approved:
                    raise ValueError("Spectrum must have valid value")

        nspecs = len(spectra)

        self._spectra = spectra
        self.nspecs = nspecs

    spectra = property(getspectra, setspectra)

    def getspec_axis(self):
        if self._spec_axis is not None:
            if self.cls.shape[self._spec_axis] != self.nspecs:
                raise ValueError("""Spectrum axis has not been updated since
                                    changing number of alm dimensions""")
        return self._spec_axis

    def setspec_axis(self, spec_axis):
        if spec_axis is not None:
            if self.cls.shape[spec_axis] != self.nspecs:
                self._spec_axis = None
                raise ValueError("""Spectrum axis does not have the right
                                    number of dimensions""")
        self._spec_axis = spec_axis

    spec_axis = property(getspec_axis, setspec_axis)

    def appendcls(self, cls, along_axis=0):
        """Add one or several cls to object instance.

        The cls must be numpy arrays or ClData objects, and along_axis
        signifies the axis along which to append the cls. If one of the cls
        has one dimension more than the other, along_axis will be interpreted
        to hold for that cls, and the 'shorter' cls will be reshaped before 
        appending. If cls is a ClData object, none of its attributes except
        lmax and the cl array will be preserved.
        """

        if isinstance(cls, ClData):
            if cls.lmax != self.lmax:
                raise ValueError("Lmax is not compatible")
            cls = cls.cls

        if cls.ndim == self.cls.ndim:
            pass
        elif cls.ndim == self.cls.ndim + 1:
            self.cls = self.cls.reshape(self.cls.shape[0:along_axis] + (1,) +
                                        self.cls.shape[along_axis:])
            if self.spec_axis is not None:
                self.spec_axis += 1
        elif cls.ndim == self.cls.ndim - 1:
            cls = cls.reshape(cls.shape[0:along_axis] + (1,) + 
                              cls.shape[along_axis:])
        else:
            raise ValueError("Incompatible number of dimensions between cls")

        if along_axis == self.cl_axis:
            raise ValueError("Cannot append along pixel axis")
        if self.spec_axis is not None:
            if along_axis == self.spec_axis:
                raise ValueError("Cannot append along spectrum axis")

        self.cls = np.append(self.cls, cls, axis=along_axis)

class _cls_iter(object):
    def __init__(self, cd):
        if not isinstance(cd, ClData):
            raise TypeError()
        self._currcls = 1
        if cd.spec_iter == True and cd.spec_axis is None:
            raise ValueError("spec_iter is True but no spec_axis given")
        self._spec_iter = cd.spec_iter
        if self._spec_iter:
            if cd.cl_axis < cd.spec_axis:
                self._subshape = list(cd.cls.shape[:cd.cl_axis] + 
                                    cd.cls.shape[cd.cl_axis + 1:cd.spec_axis] +
                                    cd.cls.shape[cd.spec_axis + 1:])
            else:
                self._subshape = list(cd.cls.shape[:cd.spec_axis] + 
                                    cd.cls.shape[cd.spec_axis + 1:cd.cl_axis] +
                                    cd.cls.shape[cd.cl_axis + 1:])
        else:
            self._subshape = list(cd.cls.shape[:cd.cl_axis] + 
                            cd.cls.shape[cd.cl_axis + 1:])
        for dim in self._subshape:
            self._currcls *= dim
        self._cls = cd.cls
        self._cl_axis = cd.cl_axis
        self._spec_axis = cd.spec_axis
        if self._spec_iter:
            if self._spec_axis < self._cl_axis:
                self._cl_axis -= 1
            else:
                self._spec_axis -= 1

        #Copies subshape
        self._currind = list(self._subshape)

    def next(self):
        if self._currcls == 0:
            raise StopIteration()
        trace_ind = len(self._subshape) - 1
        if self._currind == self._subshape:
            #First iteration
            self._currind = list(np.zeros(len(self._subshape), dtype=int))
        else:
            while (self._currind[trace_ind] == self._subshape[trace_ind] - 1):
                self._currind[trace_ind] = 0
                trace_ind -= 1
            self._currind[trace_ind] += 1
        self._currcls -= 1
        if self._spec_iter:
            if self._cl_axis < self._spec_axis:
                return self._cls[self._currind[:self._cl_axis] + [Ellipsis,]
                            + self._currind[self._cl_axis:self._spec_axis] 
                            + [Ellipsis,] + self._currind[self._spec_axis:]]
            else:
                return self._cls[self._currind[:self._spec_axis] + [Ellipsis,]
                            + self._currind[self._spec_axis:self._cl_axis] 
                            + [Ellipsis,] + self._currind[self._cl_axis:]]
        else:
            return self._cls[self._currind[:self._cl_axis] + [Ellipsis,] 
                        + self._currind[self._cl_axis:]]
