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
    def __init__(self, lmax, mmax=None, alms=None, indaxis=None,
                 polaxis=None):
        if alms is not None and indaxis is not None:
            if alms[indaxis] != lmax * (lmax + 1) // 2 + lmax + 1:
                raise ValueError("""Explicit indaxis does not contain right
                                    number of elements""")
        if indaxis is None:
            indaxis = 0
        self.indaxis = indaxis
        self._lmax = None
        if mmax != None:
            raise NotImplementedError()
        self.nnind = lmax * (lmax + 1) // 2 + lmax + 1
        if alms is None:
            alms = np.zeros(self.nnind, dtype=np.complex)
        self.lmax = lmax
        self.mmax = lmax
        self.alms = alms
        self.polaxis = polaxis

    def getalms(self):
        return self._alms

    def setalms(self, alms):
        if not isinstance(alms, np.ndarray):
            raise TypeError("Alms must be numpy array")
        if not alms.dtype == np.complex:
            raise TypeError("Alms must be complex")
        if self.indaxis >= alms.ndim or alms.shape[self.indaxis] != self.nnind:
            #Try to autodetect pixel axis
            for i in range(alms.ndim):
                if alms.shape[i] == self.nnind:
                    self.indaxis = i
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

    def getpolaxis(self):
        if self._polaxis is not None:
            if self.cls.shape[self._polaxis] != 3:
                raise ValueError("""Polarization axis has not been updated since
                                    changing number of map dimensions""")
        return self._polaxis

    def setpolaxis(self, polaxis):
        if polaxis is not None:
            if self.alms.shape[polaxis] != 3:
                self._polaxis = None
                raise ValueError("Polarization axis does not have 3 dimensions")
        self._polaxis = polaxis

    polaxis = property(getpolaxis, setpolaxis)

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

        if along_axis == self.indaxis:
            raise ValueError("Cannot append along index axis")
        if self.polaxis is not None:
            if along_axis == self.polaxis:
                raise ValueError("Cannot append along polarization axis")

        self.alms = np.append(self.alms, alms, axis=along_axis)

class ClData(object):
    def __init__(self, lmax, cls=None, spectra='temp', specaxis=None, 
                 claxis=None):
        if cls is not None and claxis is not None:
            if cls.shape[claxis] != lmax + 1:
                raise ValueError("""Explicit claxis does not contain the right
                                    number of elements""")
        if claxis is None:
            claxis = 0
        self.claxis = claxis
        self.spectra = spectra
        if cls is None:
            cls = np.zeros(lmax + 1)
        self._lmax = None
        self.lmax = lmax
        self.cls = cls
        self.specaxis = specaxis

    def getcls(self):
        return self._cls

    def setcls(self, cls):
        if not isinstance(cls, np.ndarray):
            raise TypeError("Cls must be numpy array")
        if self.claxis >= cls.ndim or cls.shape[self.claxis] != self.lmax + 1:
            #Try to autodetect cl-axis
            for i in range(cls.ndim):
                if cls.shape[i] == self.lmax + 1:
                    self.claxis = i
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

    def getspecaxis(self):
        if self._specaxis is not None:
            if self.cls.shape[self._specaxis] != self.nspecs:
                raise ValueError("""Spectrum axis has not been updated since
                                    changing number of map dimensions""")
        return self._specaxis

    def setspecaxis(self, specaxis):
        if specaxis is not None:
            if self.cls.shape[specaxis] != self.nspecs:
                self._specaxis = None
                raise ValueError("""Spectrum axis does not have the right
                                    number of dimensions""")
        self._specaxis = specaxis

    specaxis = property(getspecaxis, setspecaxis)

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
            if self.specaxis is not None:
                self.specaxis += 1
        elif cls.ndim == self.cls.ndim - 1:
            cls = cls.reshape(cls.shape[0:along_axis] + (1,) + 
                              cls.shape[along_axis:])
        else:
            raise ValueError("Incompatible number of dimensions between cls")

        if along_axis == self.claxis:
            raise ValueError("Cannot append along pixel axis")
        if self.specaxis is not None:
            if along_axis == self.specaxis:
                raise ValueError("Cannot append along spectrum axis")

        self.cls = np.append(self.cls, cls, axis=along_axis)
