from __future__ import division
import numpy as np

class CovMatData(object):
    """Class to store and pass relevant information about various types of 
        pixel-space covariance matrices. 

        Supposed to be able to store data either 'stacked' or 'interleaved'.
        Pix_axis here refers to the *first* pixel axis, and the second is
        presumed to be the following one.
    
    """
    def __init__(self, nside, ordering='ring', mat=None, pix_axis=None,
                 pol_axis=None, pol_iter=False, masked=False):
        if mat is not None and pix_axis is not None:
            if (mat.shape[pix_axis] != 12 * nside ** 2 or 
                    mat.shape[pix_axis+1] != 12 * nside ** 2):
                raise ValueError("""Explicit pix_axis does not contain the right
                                    number of pixels""")
        if pix_axis is None:
            pix_axis = 0
        self.pix_axis = pix_axis
        self._nside = None
        if mat is None:
            if not isinstance(nside, int):
                raise TypeError("nside must be an integer")
            mat = np.zeros((12 * nside ** 2, 12 * nside ** 2))
        self.nside = nside
        self.mat = mat
        self._ordering = None
        self.ordering = ordering
        self.pol_axis = pol_axis
        self.pol_iter = pol_iter

    def __iter__(self):
        return _mat_iter(self)

    def __add__(self, other):
        if _compatible(self, other):
            return CovMatData(nside=self.nside, ordering=self.ordering, 
                           pix_axis=self.pix_axis, pol_axis=self.pol_axis,
                           pol_iter=self.pol_iter, mat=self.mat + other.mat)
        else:
            raise ValueError("Matrices not compatible for adding")

    #Rewrite this to matrix multiplication
    def __mul__(self, other):
        pass
#        if _compatible(self, other):
#            return CovMatData(nside=self.nside, ordering=self.ordering, 
#                           pix_axis=self.pix_axis, pol_axis=self.pol_axis,
#                           pol_iter=self.pol_iter, mat=self.mat * other.mat)
#        else:
#            raise ValueError("Mats not compatible for multiplying")

    def __sub__(self, other):
        if _compatible(self, other):
            return CovMatData(nside=self.nside, ordering=self.ordering, 
                           pix_axis=self.pix_axis, pol_axis=self.pol_axis,
                           pol_iter=self.pol_iter, mat=self.mat - other.mat)
        else:
            raise ValueError("Covariance matrices not compatible for subtracting")

    #Rewrite this to multiplication by inverse
    def __truediv__(self, other):
        pass
#        if _compatible(self, other):
#            return MapData(nside=self.nside, ordering=self.ordering, 
#                           pix_axis=self.pix_axis, pol_axis=self.pol_axis,
#                           pol_iter=self.pol_iter, map=self.map / other.map)
#        else:
#            raise ValueError("Maps not compatible for dividing")

    #Not sure if this is the best way to implement this function, i.e. making 
    #a copy, so commenting it out for the time being
    def __getitem__(self, index):
        pass
#        n = MapData(nside=self.nside, ordering=self.ordering, 
#                       pol_iter=self.pol_iter, map=self.map[index])
#        if n.pix_axis == self.pix_axis or self.pol_axis is None:
#            n.pol_axis = self.pol_axis
#        else:
#            n.pol_axis = None
#        return n

    def getmat(self):
        return self._mat

    def setmat(self, mat):
        if not isinstance(mat, np.ndarray):
            raise TypeError("Mat must be numpy array")
        if (self.pix_axis >= mat.ndim or 
                mat.shape[self.pix_axis] != 12*self.nside**2):
            #Try to autodetect pixel axis
            for i in range(mat.ndim):
                if (mat.shape[i] == 12*self.nside**2 and 
                        mat.shape[i + 1] == 12 * self.nside ** 2):
                    self.pix_axis = i
                    break
            else:
                raise ValueError("""Pixel number of input covariance matrix 
                                    does not conform to nside""")
        self._mat = mat

    mat = property(getmat, setmat)

    def getordering(self):
        return self._ordering
    
    def setordering(self, ordering):
        #For the time being, once ordering is given, not possible to change
        #for a covariance matrix
        if self._ordering is not None:
            raise ValueError("ordering is immutable")
        if not isinstance(ordering, str):
            raise TypeError("Ordering must be a string")
        if ordering.lower() != 'ring' and ordering.lower() != 'nested':
            raise ValueError("Ordering must be ring or nested")
        self._ordering = ordering.lower()

    ordering = property(getordering, setordering)

    def getnside(self):
        return self._nside

    def setnside(self, nside):
        if self._nside is not None:
            raise ValueError("nside is immutable")
        if not isinstance(nside, int):
            raise TypeError("nside must be an integer")
        b = bin(nside)[2:]
        if (b[0] != '1' or int(b[1:], 2) != 0):
            raise ValueError('nside has invalid value')
        self._nside = nside

    nside = property(getnside, setnside)

    def getpol_axis(self):
        if self._pol_axis is not None:
            if self.mat.shape[self._pol_axis] != 3:
                raise ValueError("""Polarization axis has not been updated since
                                    changing number of covmat dimensions""")
        return self._pol_axis

    def setpol_axis(self, pol_axis):
        if pol_axis is not None:
            if self.mat.shape[pol_axis] != 3:
                self._pol_axis = None
                raise ValueError("Polarization axis does not have 3 dimensions")
        self._pol_axis = pol_axis

    pol_axis = property(getpol_axis, setpol_axis)

    def appendmats(self, mat, along_axis=0):
        """Add one or several covariance matricess to object instance.

        The covariance matrices must be numpy arrays or CovMatData objects, 
        and along_axis
        signifies the axis along which to append the covariance matrix. If one
        of the covariance matrices
        has one dimension more than the other, along_axis will be interpreted
        to hold for that covariance matrix, and the 'shorter' matrix will be 
        reshaped before appending.
        """

        if isinstance(mat, CovMatData):
            if mat.nside != self.nside:
                raise ValueError("Nside is not compatible")
            mat = mat.mat

        if mat.ndim == self.mat.ndim:
            pass
        elif mat.ndim == self.mat.ndim + 1:
            self.mat = self.mat.reshape(self.mat.shape[0:along_axis] + (1,) +
                                        self.mat.shape[along_axis:])
        elif mat.ndim == self.mat.ndim - 1:
            mat = mat.reshape(mat.shape[0:along_axis] + (1,) + 
                              mat.shape[along_axis:])
        else:
            raise ValueError("""Incompatible number of dimensions between 
                                covariance matrices""")

        if along_axis == self.pix_axis or along_axis == self.pix_axis + 1:
            raise ValueError("Cannot append along pixel axis")
        if self.pol_axis is not None:
            if along_axis == self.pol_axis:
                raise ValueError("Cannot append along polarization axis")

        self.mat = np.append(self.mat, mat, axis=along_axis)

class _mat_iter(object):
    def __init__(self, md):
        if not isinstance(md, CovMatData):
            raise TypeError()
        self._currmat = 1
        if md.pol_iter == True and md.pol_axis is None:
            raise ValueError("pol_iter is True but no pol_axis given")
        self._pol_iter = md.pol_iter
        if self._pol_iter:
            if md.pix_axis < md.pol_axis:
                self._subshape = list(md.mat.shape[:md.pix_axis] + 
                                    md.mat.shape[md.pix_axis + 2:md.pol_axis] +
                                    md.mat.shape[md.pol_axis + 1:])
            else:
                self._subshape = list(md.mat.shape[:md.pol_axis] + 
                                    md.mat.shape[md.pol_axis + 1:md.pix_axis] +
                                    md.mat.shape[md.pix_axis + 2:])
        else:
            self._subshape = list(md.mat.shape[:md.pix_axis] + 
                            md.mat.shape[md.pix_axis + 2:])
        for dim in self._subshape:
            self._currmat *= dim
        self._mat = md.mat
        self._pix_axis = md.pix_axis
        self._pol_axis = md.pol_axis
        if self._pol_iter:
            if self._pol_axis < self._pix_axis:
                self._pix_axis -= 1
            else:
                self._pol_axis -= 1
        #Copies subshape
        self._currind = list(self._subshape)

    def next(self):
        if self._currmat == 0:
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
        self._currmat -= 1
        if self._pol_iter:
            if self._pix_axis < self._pol_axis:
                return self._mat[self._currind[:self._pix_axis] + [Ellipsis,]
                            + self._currind[self._pix_axis:self._pol_axis] 
                            + [Ellipsis,] + self._currind[self._pol_axis:]]
            else:
                return self._mat[self._currind[:self._pol_axis] + [Ellipsis,]
                            + self._currind[self._pol_axis:self._pix_axis] 
                            + [Ellipsis,] + self._currind[self._pix_axis:]]
        else:
            return self._mat[self._currind[:self._pix_axis] + [Ellipsis,] 
                        + self._currind[self._pix_axis:]]
