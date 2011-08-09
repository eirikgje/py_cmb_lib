from __future__ import division
import numpy as np
import sys
#import fileutils

_r2n = {}
_n2r = {}
_pix2x = None
_pix2y = None
_x2pix = None
_y2pix = None

def _init_r2n(nside):
    global _r2n
    global _x2pix
    global _y2pix
    if _x2pix is None:
        _mk_xy2pix()
    
    jrll = np.array((2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4))
    jpll = np.array((1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7))
    npix = 12*nside**2
    pixs = np.arange(1,npix+1)
    nl2 = 2*nside
    nl4 = 4*nside
    ncap = nl2*(nside-1)
    #hip = np.zeros(npix)
    #fihip = np.zeros(npix)
    #irn = np.zeros(npix)
    #iphi = np.zeros(npix)
    #nr = np.zeros(npix)
    #face_num = np.zeros(npix)

    #South polar cap (default)
    ip = npix - pixs + 1
    hip = ip / 2.0
    fihip = hip // 1
    irs = (np.sqrt(hip - np.sqrt(fihip))).astype(int) + 1
    iphi = 4 * irs + 1 - (ip - 2 * irs * (irs-1))
    kshift = np.zeros(npix, int)
    nr = irs
    irn = nl4 - irs 
    face_num = (iphi - 1) // irs + 8

    #Equatorial region
    filter = pixs <= nl2 * (5 * nside + 1)
    ip[filter] = pixs[filter] - ncap - 1
    irn[filter] = (ip[filter] / nl4).astype(int) + nside
    iphi[filter] = ip[filter] % nl4 + 1
    kshift[filter] = (irn[filter] + nside) % 2
    nr[filter] = nside
    ire = irn[filter] - nside + 1
    irm = nl2 + 2 - ire
    ifm = (iphi[filter] - ire // 2 + nside - 1) // nside
    ifp = (iphi[filter] - irm // 2 + nside - 1) // nside
    face_num[filter[filter] & (ifp - 1 == ifm)] = ifp[ifp - 1 == ifm] + 7
    face_num[filter[filter] & (ifp + 1 == ifm)] = ifp[ifp + 1 == ifm]
    face_num[filter[filter] & (ifp == ifm)] = ifp[ifp == ifm] % 4 + 4

    #North polar cap
    filter = pixs <= ncap
    hip[filter] = pixs / 2.0
    fihip[filter] = hip[filter] // 1
    irn[filter] = (np.sqrt(hip[filter] - np.sqrt(fihip[filter]))).astype(int) + 1
    iphi[filter] = pixs[filter] - 2 * irn[filter] * (irn[filter] - 1)
    nr[filter] = irn[filter]
    face_num[filter] = (iphi[filter] - 1) // irn[filter]

    irt = irn - jrll[face_num] * nside + 1
    ipt = 2 * iphi - jpll[face_num] * nr - kshift - 1
    ipt[ipt > nl2] = ipt[ipt > nl2] - 8 * nside
    ix = (ipt - irt) // 2
    iy = -(ipt + irt) // 2
    ix_low = ix % 128
    ix_hi = ix // 128
    iy_low = iy % 128
    iy_hi = iy // 128

    ipf = ((_x2pix[ix_hi] + _y2pix[iy_hi]) * (128 * 128) + 
            _x2pix[ix_low] + _y2pix[iy_low])

    _r2n[nside] = ipf + face_num * nside ** 2

def _mk_xy2pix():
    global _x2pix
    global _y2pix
    _x2pix = np.zeros(128, int)
    _y2pix = np.zeros(128, int)
    if sys.version[0:3] >= '2.6':
        for i in range(128):
            b = bin(i)[2:]
            _x2pix[i] = int(b, 4)
            _y2pix[i] = 2 * int(b, 4)
    else:
        import binmod
        for i in range(128):
            b = binmod.bin(i)
            _x2pix[i] = int(b, 4)
            _y2pix[i] = 2 * int(b, 4)

def _init_n2r(nside):
    global _n2r
    global _pix2x
    global _pix2y
    npix = 12 * nside * nside
    #_n2r[nside] = np.zeros(npix)
    #For now: Naive almost direct implementation of the nest2ring source code
    #(but using list comprehensions instead of loops)
    jrll = np.array((2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4))
    jpll = np.array((1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7))
    if _pix2x is None:
        _mk_pix2xy()
    ncap = 2 * nside * (nside-1)
    nl4 = 4 * nside
    npface = nside * nside

    pixs = np.arange(npix)
    facenum = pixs // npface
    ipf = pixs % npface
    ip_low = ipf % 1024
    ip_trunc = ipf // 1024
    ip_med = ip_trunc % 1024
    ip_hi = ip_trunc // 1024
    ix = 1024*_pix2x[ip_hi] + 32*_pix2x[ip_med] + _pix2x[ip_low]
    iy = 1024*_pix2y[ip_hi] + 32*_pix2y[ip_med] + _pix2y[ip_low]
    jrt = ix + iy
    jpt = ix - iy
    jr = jrll[facenum] * nside - jrt - 1

    nr = np.zeros(npix, int)
    n_before = np.zeros(npix, int)
    kshift = np.zeros(npix, int)

    #Equatorial region
    nr[:] = nside
    n_before = ncap + nl4 * (jr - nside)
    kshift = (jr-nside)%2

    #South pole
    nr[jr > 3 * nside] = nl4 - jr[jr > 3 * nside]
    n_before[jr > 3 * nside] = (npix - 2 * (nr[jr > 3 * nside] + 1) * 
                                nr[jr > 3 * nside])
    kshift[jr > 3 * nside] = 0

    #North pole
    nr[jr < nside] = jr[jr < nside]
    n_before[jr < nside] = 2 * nr[jr < nside] * (nr[jr < nside] - 1)
    kshift[jr < nside] = 0

    jp = (jpll[facenum] * nr + jpt + 1 + kshift) // 2
    jp[jp > nl4] = jp[jp > nl4] - nl4
    jp[jp < 1] = jp[jp < 1] + nl4

    _n2r[nside] = n_before + jp - 1

def _mk_pix2xy():
    global _pix2x
    global _pix2y
    _pix2x = np.zeros(1024, int)
    _pix2y = np.zeros(1024, int)

    #pix2x contains the sum of all odd bits, pix2y all the even ones.
    if sys.version[0:3] >= '2.6':
        for i in range(1024):
            b = bin(i)[2:]
            _pix2x[i] = int(b[-1::-2], 2)
            if len(b) == 1:
                _pix2y[i] = 0
            else:
                _pix2y[i] = int(b[-2::-2], 2)
    else:
        import binmod
        for i in range(1024):
            b = binmod.bin(i)
            _pix2x[i] = int(b[-1::-2], 2)
            if len(b) == 1:
                _pix2y[i] = 0
            else:
                _pix2y[i] = int(b[-2::-2], 2)

def degrade_average(mapd, nside_n):
    """Degrade input map to nside resolution by averaging over pixels.
    
    Output map will be in nested ordering, no matter what the input map was.
    """
    if mapd.ordering == 'ring':
        mapd.switchordering()

    redfact = (mapd.nside/nside_n)**2
    mapd.map = np.reshape(mapd.map, (12*nside_n*nside_n, redfact))
    mapd.map = np.average(mapd.map, axis=1)
    mapd.nside = nside_n

def degrade(mapd, nside_n, pixwin=None):
    if pixwin is None:
        degrade_average(mapd, nside_n)

def ring2nest(map, nside):
    """Assumes map has shape (nmaps, npix)"""
    global _r2n

    if sys.version[0:3] >= '2.6':
        b = bin(nside)[2:]
    else:
        import binmod
        b = binmod.bin(nside)
    if (b[0] != '1' or int(b[1:],2) !=0):
        raise ValueError('nest2ring: nside has invalid value')

    if not _r2n.has_key(nside):
        _init_r2n(nside)
    return map[:, _r2n[nside]]

def nest2ring(map, nside):
    """Assumes map has shape (nmaps, npix)"""
    global _n2r

    if sys.version[0:3] >= '2.6':
        b = bin(nside)[2:]
    else:
        import binmod
        b = binmod.bin(nside)
    if (b[0] != '1' or int(b[1:], 2) != 0):
        raise ValueError('nest2ring: nside has invalid value')

    if not _n2r.has_key(nside):
        _init_n2r(nside)
    return map[:, _n2r[nside]]

class MapData(object):
    """Class to store and pass relevant map information.

    The basic map structure is (nmaps, npix). Assignments with a
    one-dimensional array will reshape to this form. Ordering should be set
    once, subsequently it should be switched with switchordering().

    """
#    It is possible to 
#    define subdivisions of this structure; these will all be 'static' 
#    in the sense that if md 
#    is a MapData instance, md.subdivide(3) (e.g. for polarization) will
#    require that in assignments (md.map = map) or appends (md.append(map)), 
#    map must be either (3, nmaps, npix) or (3, npix). nmaps is thus the only
#    'dynamic' dimension. Further subdivisions are possible (see
#    Mapdata.subdivide() for more information.)
#    """
    #Will contain the data needed to fully describe a HEALPix map,
    #with read and write statements.
    def __init__(self):
        self._map = None
        self._ordering = None
        self.nside = None
        self.subd = None
        self.dyn_ind = 0

    def getmap(self):
        return self._map

    def setmap(self, map):
        map = self.conform_map(map)
        self._map = map

    map = property(getmap, setmap)

    def getordering(self):
        return self._ordering
    
    def setordering(self, ordering):
        if ordering.lower() != 'ring' and ordering.lower() != 'nested':
            raise ValueError("Ordering must be ring or nested")
        self._ordering = ordering.lower()

    ordering = property(getordering, setordering)

    def switchordering(self):
        if self.ordering is None:
            raise ValueError("No ordering given for map")
        if self.map is not None:
            if self.nside is None:
                raise ValueError("No nside given for map")
            if len(self.map[-1,:]) != 12*self.nside**2:
                raise ValueError("Number of pixels incompatible with nside")
        if self.ordering == 'ring':
            if self.map is not None:
                self.map = ring2nest(self.map, self.nside)
            self.ordering = 'nested'
        elif self.ordering == 'nested':
            if self.map is not None:
                self.map = nest2ring(self.map, self.nside)
            self.ordering='ring'

    def subdivide(self, vals):
        """Can take either int, tuple or numpy arrays as arguments.
        
        By default, subdividing will be done in such a way that the dynamical
        index (i.e. number of map samples or whatever) is the next-to-last one
        (the last one being the number of map pixels). Note that adding samples
        should have nothing to do with subdividing - subdividing is for those
        cases where each sample will have more than one map (polarization, 
        various frequencies etc.) Also note that after subdividing, each
        map array added (or assigned) to the MapData instance must have the
        shape (x1, x2, ..., xn, n, npix) or (x1, x2, ..., xn, npix) where
        x1, x2, ..., xn are the subdivisions and n is the number of map samples
        added (could be 1 and will be interpreted as 1 if missing). 
        Subdivision can be done several times. The new dimension will then
        be the leftmost dimension in the resulting MapData.map array.

        """
        if isinstance(vals, int):
            self.dyn_ind += 1
        elif isinstance(vals, tuple) or isinstance(vals, np.ndarray):
            self.dyn_ind += np.size(vals)
        else:
            raise TypeError('Must be int, tuple or np.ndarray')

        if self.subd is None:
            self.subd = np.array(vals)
        else:
            self.subd = np.append(vals, self.subd)

        if self.map is not None:
            self._map = np.resize(self._map, np.append(self.subd, 
                                    (np.size(self._map, -2), 
                                    np.size(self._map, -1))))

    def appendmaps(self, map):
        """Add one or several maps to object instance.

        The input map(s) must be numpy arrays, and they must have the shape
        (subd, nmaps, npix) or (subd, npix) where subd is the current
        subdivision of the map instance, npix is the number of pixels of the
        maps already added to the object instance. If there currently are no
        maps added to the instance, this function is equivalent to an
        assignment (and npix can be whatever). nmaps can be any number, and if
        this dimension is missing from the array, it will be interpreted as a
        single map. If there are no subdivisions, a (npix) numpy array is
        acceptable.

        """
        if self.map is None:
            self.map = map
        else:
            if np.size(map, -1) != np.size(self.map, -1):
                raise ValueError("Incorrect number of pixels in input map")
            map = self.conform_map(map)
            self.map = np.append(self.map, map, axis=self.dyn_ind)

    def conform_map(self, map):
        """Make input map acceptable shape, or raise exception.
        
        Input map is only compared to the current subdivision, not any present
        maps.
        
        """
        if not isinstance(map, np.ndarray):
            raise TypeError('Map must be numpy array')
        if self.subd == None:
            if map.ndim == 1:
                map = map.reshape((1, np.size(map)))
            elif map.ndim > 2:
                raise ValueError('Too many dimensions in map')
            elif map.ndim < 1:
                raise ValueError('Too few dimensions in map')
        else:
            mlen = np.size(self.subd) + 2
            if mlen > map.ndim + 1:
                raise ValueError('Too few dimensions in map')
            elif mlen < map.ndim:
                raise ValueError('Too many dimensions in map')
            if mlen == map.ndim:
                if any(np.shape(map)[:-2] != self.subd):
                    raise ValueError("""Map dimensions do not conform to MapData
                                    subdivision""")
            elif mlen == map.ndim + 1:
                if (any(np.shape(map)[:-1] != self.subd)):
                    raise ValueError("""Map dimensions do not conform to MapData
                                    subdivision""")
                else:
                    map = map.reshape((np.append(self.subd, 
                                        (1, np.size(map, -1)))))
        return map

    #def write(self, filename, ftype='fits'):
    #    try:
    #        self.validate():
    #    except ValidateError as 
    #        if ftype='fits':
    #            fileutils.write_fitsmap(filename, self)
    #        elif ftype='npy':
    #            np.save(filename, self)
    #def validate(self):
    #    if self.map is None or self.ordering is None:
    #        return False
    #    if self.npix is None:
    #        self.npix = np.size(self.map, axis=0)
    #    if self.nside is None:
    #        self.nside = np.int(np.sqrt(self.npix//12))
    #    return self.npix == 12*self.nside*self.nside
