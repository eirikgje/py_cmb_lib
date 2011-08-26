from __future__ import division
import numpy as np
import sys
from versionmod import bin, any, all
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
    npix = 12 * nside ** 2
    #If the other is already initialized, use that information instead
    if _n2r.has_key(nside):
        _r2n[nside] = _n2r[nside].argsort()
        return
    if _x2pix is None:
        _mk_xy2pix()
    
    pixs = np.arange(npix)
    jrll = np.array((2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4))
    jpll = np.array((1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7))
    nl2 = 2 * nside
    nl4 = 4 * nside
    ncap = nl2 * (nside - 1)
    
    #irn: Ring number, counted from north pole
    #irs: Ring number, counted from south pole
    ip = np.zeros(npix, int)
    irs = np.zeros(npix, int)
    iphi = np.zeros(npix, int)
    nr = np.zeros(npix, int)
    irn = np.zeros(npix, int)
    face_num = np.zeros(npix, int)
    kshift = np.zeros(npix, int)

    #South polar cap 
    filter = pixs >= npix - ncap 
    ip[filter] = npix - pixs[filter]
    irs[filter] = (np.sqrt(ip[filter] * 0.5).round()).astype(int) 
    iphi[filter] = 2 * irs[filter] * (irs[filter] + 1) - ip[filter]
    irs[filter], iphi[filter] = _correct_ring_phi(1, irs[filter], iphi[filter])
    nr[filter] = irs[filter]
    irn[filter] = nl4 - irs[filter] 
    face_num[filter] = iphi[filter] // irs[filter] + 8 #in {0, 11}
    kshift[filter] = 0

    #Equatorial region
    filter = (pixs < npix - ncap) & (pixs >= ncap)
    ip[filter] = pixs[filter] - ncap
    irn[filter] = ip[filter] // nl4 + nside 
    iphi[filter] = ip[filter] % nl4  
    # 1 if irn+nside is odd, 0 otherwise
    kshift[filter] = (irn[filter] + nside) % 2
    nr[filter] = nside
    ire = np.zeros(npix, int)
    irm = np.zeros(npix, int)
    ifm = np.zeros(npix, int)
    ifp = np.zeros(npix, int)
    ire[filter] = irn[filter] - nside + 1 # in {1, 2*nside +1}
    irm[filter] = nl2 + 2 - ire[filter]
    # face boundary
    ifm[filter] = (iphi[filter] - ire[filter] // 2 + nside) // nside
    ifp[filter] = (iphi[filter] - irm[filter] // 2 + nside) // nside
    # (half-)faces 8 to 11
    face_num[filter & (ifp > ifm)] = ifp[filter & (ifp > ifm)] + 7 
    # (half-)faces 0 to 3
    face_num[filter & (ifp < ifm)] = ifp[filter & (ifp < ifm)] 
    # faces 4 to 7
    face_num[filter & (ifp == ifm)] = (ifp[filter & (ifp == ifm)] & 3) + 4

    #North polar cap
    filter = pixs < ncap
    irn[filter] = (np.sqrt((pixs[filter] + 1) * 0.5).round()).astype(int)
    iphi[filter] = pixs[filter] - 2 * irn[filter] * (irn[filter] - 1)
    irn[filter], iphi[filter] = _correct_ring_phi(1, irn[filter], iphi[filter])
    nr[filter] = irn[filter]
    face_num[filter] = iphi[filter] // irn[filter]
    kshift[filter] = 0

    irt = irn - jrll[face_num] * nside + 1
    ipt = 2 * iphi - jpll[face_num] * nr - kshift + 1
    ipt[ipt >= nl2] = ipt[ipt >= nl2] - 8 * nside
    ix = (ipt - irt) // 2
    iy = -(ipt + irt) // 2
    ix_low = ix % 128
    ix_hi = ix // 128
    iy_low = iy % 128
    iy_hi = iy // 128

    ipf = ((_x2pix[ix_hi] + _y2pix[iy_hi]) * (128 * 128) + 
            _x2pix[ix_low] + _y2pix[iy_low])

    _r2n[nside] = ipf + face_num * nside ** 2

def _correct_ring_phi(location, iring, iphi):
    delta = np.zeros(len(iphi), int)
    delta[iphi < 0] += 1
    delta[iphi >= 4 * iring] -= 1
    iring[delta != 0] = iring[delta != 0] - location * delta[delta != 0]
    iphi[delta != 0] = iphi[delta != 0] + delta[delta != 0] * (4
                                                        * iring[delta != 0])
    return iring, iphi

def _mk_xy2pix():
    global _x2pix
    global _y2pix
    _x2pix = np.zeros(128, int)
    _y2pix = np.zeros(128, int)
    for i in range(128):
        b = bin(i)[2:]
        _x2pix[i] = int(b, 4)
        _y2pix[i] = 2 * int(b, 4)

def _init_n2r(nside):
    global _n2r
    global _pix2x
    global _pix2y
    if _r2n.has_key(nside):
        _n2r[nside] = _r2n[nside].argsort()
        return

    npix = 12 * nside * nside
    #For now: Naive almost direct implementation of the nest2ring source code
    #(but using list comprehensions instead of loops)
    jrll = np.array((2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4))
    jpll = np.array((1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7))
    if _pix2x is None:
        _mk_pix2xy()
    nl4 = 4 * nside
    npface = nside * nside

    pixs = np.arange(npix)
    facenum = pixs // npface
    ipf = pixs % npface
    ip_low = ipf % 1024
    ip_trunc = ipf // 1024
    ip_med = ip_trunc % 1024
    ip_hi = ip_trunc // 1024
    ix = 1024 * _pix2x[ip_hi] + 32 * _pix2x[ip_med] + _pix2x[ip_low]
    iy = 1024 * _pix2y[ip_hi] + 32 * _pix2y[ip_med] + _pix2y[ip_low]
    jrt = ix + iy
    jpt = ix - iy
    jr = jrll[facenum] * nside - jrt - 1

    nr = np.zeros(npix, int)
    n_before = np.zeros(npix, int)
    kshift = np.zeros(npix, int)

    #Equatorial region
    nr[:] = nside
    n_before = 2 * nr * (2 * jr - nr - 1)
    kshift = (jr - nside) % 2

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

    #pix2x contains the integer repr. of all odd bits, pix2y all the even ones.
    for i in range(1024):
        b = bin(i)[2:]
        _pix2x[i] = int(b[-1::-2][::-1], 2)
        if len(b) == 1:
            _pix2y[i] = 0
        else:
            _pix2y[i] = int(b[-2::-2][::-1], 2)

def degrade_average(mapd, nside_n):
    """Degrade input map to nside resolution by averaging over pixels.
    
    """
    switched = False
    if mapd.ordering == 'ring':
        switched = True
        mapd.switchordering()

    redfact = (mapd.nside//nside_n)**2
    temp = np.reshape(mapd.map, np.append(mapd.map.shape[:-1], 
                        (12*nside_n*nside_n, redfact)))
    mapd.map = np.average(temp, axis=-1)
    mapd.nside = nside_n
    if switched: mapd.switchordering()
    return mapd

def degrade(mapd, nside_n, pixwin=None):
    if pixwin is None:
        return degrade_average(mapd, nside_n)

def ring2nest(map, nside):
    global _r2n
    b = bin(nside)[2:]
    if (b[0] != '1' or int(b[1:],2) != 0):
        raise ValueError('nest2ring: nside has invalid value')

    if not _r2n.has_key(nside):
        _init_r2n(nside)
    return map[..., _r2n[nside]]

def nest2ring(map, nside):
    global _n2r

    b = bin(nside)[2:]
    if (b[0] != '1' or int(b[1:], 2) != 0):
        raise ValueError('nest2ring: nside has invalid value')

    if not _n2r.has_key(nside):
        _init_n2r(nside)
    return map[..., _n2r[nside]]

def ring2nest_ind(ind, nside):
    global _r2n

    b = bin(nside)[2:]
    if (b[0] != '1' or int(b[1:], 2) != 0):
        raise ValueError('ring2nest_ind: nside has invalid value')
    if not _r2n.has_key(nside):
        _init_r2n(nside)

    return _r2n[nside][ind]

def nest2ring_ind(ind, nside):
    global _n2r

    b = bin(nside)[2:]
    if (b[0] != '1' or int(b[1:], 2) != 0):
        raise ValueError('nest2ring_ind: nside has invalid value')

    if not _n2r.has_key(nside):
        _init_n2r(nside)

    return _n2r[nside][ind]


class MapData(object):
    """Class to store and pass relevant HEALPix map information.

    The basic map structure is (nmaps, npix). Assignments with a
    one-dimensional array will reshape to this form. Ordering should be set
    at construction, subsequently it should be switched with switchordering().
    At this point, nside is uniquely determined by the last dimension of the
    map array, and all that is needed to initialize a map is an nside. This
    will let the MapData.map array be an array of zeroes.

    """
    def __init__(self, nside, ordering='ring', map=None, lsubd=None,
                 rsubd=None, pol=False):
        self.dyn_ind = 0
        self._map = None
        self.subd = ()
        self.pol = pol
        if lsubd is not None:
            self.subdivide(lsubd)
        if rsubd is not None:
            self.subdivide(rsubd, left_of_dyn_d=False)
        if map is None:
            map = np.zeros(self.subd[0:self.dyn_ind] + (1,) + 
                           self.subd[self.dyn_ind:] + (12*nside**2,))
        self.map = map
        self.nside = nside
        self.ordering = ordering

    def getmap(self):
        return self._map

    def setmap(self, map):
        self._map = self.conform_map(map)

    map = property(getmap, setmap)

    def getordering(self):
        return self._ordering
    
    def setordering(self, ordering):
        if ordering is None:
            self._ordering = ordering
        else:
            if ordering.lower() != 'ring' and ordering.lower() != 'nested':
                raise ValueError("Ordering must be ring or nested")
            self._ordering = ordering.lower()

    ordering = property(getordering, setordering)

    def getnside(self):
        return self._nside

    def setnside(self, nside):
        if not isinstance(nside, int):
            raise TypeError("nside must be an integer")
        else:
            if np.size(self.map, -1) != 12*nside*nside:
                raise ValueError("""nside must be compatible with last
                                    dimension of map""")
            self._nside = nside

    nside = property(getnside, setnside)

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


    def switchordering(self):
        if self.ordering == 'ring':
            if self.map is not None:
                self.map = ring2nest(self.map, self.nside)
            self.ordering = 'nested'
        elif self.ordering == 'nested':
            if self.map is not None:
                self.map = nest2ring(self.map, self.nside)
            self.ordering='ring'

    def subdivide(self, vals, left_of_dyn_d=True):
        """Can take either int, tuple or numpy arrays as arguments.
        
        By default, subdividing will be done in such a way that the dynamical
        index (i.e. number of map samples or whatever) is the next-to-last one
        (the last one being the number of map pixels). Note that adding samples
        should have nothing to do with subdividing - subdividing is for those
        cases where each sample will have more than one map (polarization, 
        various frequencies etc.) Note, however, that for polarization it is
        possible to just set the 'pol' keyword to True for the object, and the 
        subdivision will be taken care of. Also note that after subdividing, 
        each map array added (or assigned) to the MapData instance must have the
        shape (x1, x2, ..., xn, n, npix) or (x1, x2, ..., xn, npix) where
        x1, x2, ..., xn are the subdivisions and n is the number of map samples
        added (could be 1 and will be interpreted as 1 if missing). 
        Subdivision can be done several times. The new dimension will then
        be the leftmost dimension in the resulting MapData.map array.

        """
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

        if self.map is not None:
            self._map = np.resize(self.map, self.subd[0:self.dyn_ind] +
                                  (self.map.shape[old_dyn_ind],) +
                                  self.subd[self.dyn_ind:] +
                                  (self.map.shape[-1],))

    def appendmaps(self, map):
        """Add one or several maps to object instance.

        The input map(s) must be numpy arrays, or MapData objects
        and they must have the shape
        (subd, nmaps, npix) or (subd, npix) where subd is the current
        subdivision of the MapData instance, npix is the number of pixels of the
        maps already added to the object instance. nmaps can be any number, 
        and if this dimension is missing from the array, it will be interpreted 
        as a single map. If there are no subdivisions, a (npix) numpy array is
        acceptable.

        """
        if isinstance(map, MapData):
            if map.nside != self.nside:
                raise ValueError("Nside is not compatible")
            map = map.map
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
        mlen = len(self.subd) + 2
        if mlen > map.ndim + 1:
            raise ValueError('Too few dimensions in map')
        elif mlen < map.ndim:
            raise ValueError('Too many dimensions in map')
        if mlen == map.ndim:
            #Explicit dynamic dimension
            mapsubd = (map.shape[0:self.dyn_ind] + 
                        map.shape[self.dyn_ind + 1:-1])
            if (mapsubd != self.subd):
                print mapsubd
                print self.subd
                raise ValueError("""Map dimensions do not conform to MapData
                                subdivision""")
        elif mlen == map.ndim + 1:
            #Dynamic dimension is implicit
            mapsubd = (map.shape[0:-1])
            if (mapsubd  != self.subd):
                raise ValueError("""Map dimensions do not conform to MapData
                                subdivision""")
            else:
                map = map.reshape(self.subd[0:self.dyn_ind] + (1,) + 
                                  self.subd[self.dyn_ind:] + 
                                  (np.size(map, -1),))
        return map
