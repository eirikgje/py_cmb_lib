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

def degrade_average(md, nside_n):
    """Degrade input map to nside resolution by averaging over pixels.
    
    """
    switched = False
    if md.ordering == 'ring':
        switched = True
        md.switchordering()

    redfact = (md.nside // nside_n) ** 2
    temp = np.reshape(md.map, md.map.shape[:md.pixaxis] + 
                      (12*nside_n*nside_n, redfact) + 
                      md.map.shape[md.pixaxis + 1:])
    #nmap = MapData(nside_n, ordering='nested')
    md._nside = nside_n
    md.map = np.average(temp, axis=md.pixaxis + 1)
    if switched: md.switchordering()
    return md

def degrade(md, nside_n, pixwin=None):
    if pixwin is None:
        return degrade_average(md, nside_n)
    else:
        raise NotImplementedError()

def ring2nest(md):
    global _r2n

    if not _r2n.has_key(md.nside):
        _init_r2n(md.nside)
    md.map = md.map[(Ellipsis,) * md.pixaxis + (_r2n[md.nside],) + 
                    (Ellipsis,) * (md.map.ndim - 1 - md.pixaxis)]
    md.ordering='nested'

    return md

def nest2ring(md):
    global _n2r

    if not _n2r.has_key(md.nside):
        _init_n2r(md.nside)

    md.map = md.map[(Ellipsis,) * md.pixaxis + (_n2r[md.nside],) + 
                    (Ellipsis,) * (md.map.ndim - 1 - md.pixaxis)]
    md.ordering='ring'
    return md

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
    will make the MapData.map array be an array of zeroes. nside (and npix)
    for a map will be practically immutable after construction.

    """
    def __init__(self, nside, ordering='ring', map=None, pixaxis=None,
                 rsubd=None, pol_axis=None):
        if map is not None and pixaxis is not None:
            if map[pixaxis] != 12*nside**2:
                raise ValueError("""Explicit pixaxis does not contain the right
                                    number of pixels""")
        if pixaxis is None:
            pixaxis = 0
        self.pixaxis = pixaxis
        self._nside = None
        if map is None:
            if not isinstance(nside, int):
                raise TypeError("nside must be an integer")
            map = np.zeros(12*nside**2)
        self.nside = nside
        self.map = map
        self.ordering = ordering
        self.pol_axis = pol_axis

    def getmap(self):
        return self._map

    def setmap(self, map):
        if not isinstance(map, np.ndarray):
            raise TypeError("Map must be numpy array")
        if (self.pixaxis >= map.ndim or 
                map.shape[self.pixaxis] != 12*self.nside**2):
            #Try to autodetect pixel axis
            for i in range(map.ndim):
                if map.shape[i] == 12*self.nside**2:
                    self.pixaxis = i
                    break
            else:
                raise ValueError("""Pixel number of input map does not conform 
                                    to nside""")
        self._map = map
        #Will raise an error if the new map does not have the correct number of
        #elements along pol_axis

    map = property(getmap, setmap)

    def getordering(self):
        return self._ordering
    
    def setordering(self, ordering):
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
            if self.map.shape[self._pol_axis] != 3:
                raise ValueError("""Polarization axis has not been updated since
                                    changing number of map dimensions""")
        return self._pol_axis

    def setpol_axis(self, pol_axis):
        if pol_axis is not None:
            if self.map.shape[pol_axis] != 3:
                self._pol_axis = None
                raise ValueError("Polarization axis does not have 3 dimensions")
        self._pol_axis = pol_axis

    pol_axis = property(getpol_axis, setpol_axis)

    def switchordering(self):
        if self.ordering == 'ring':
            self = ring2nest(self)
        elif self.ordering == 'nested':
            self = nest2ring(self)

    def appendmaps(self, map, along_axis=0):
        """Add one or several maps to object instance.

        The maps must be numpy arrays or MapData objects, and along_axis
        signifies the axis along which to append the map. If one of the maps
        has one dimension more than the other, along_axis will be interpreted
        to hold for that map, and the 'shorter' map will be reshaped before 
        appending.
        """

        if isinstance(map, MapData):
            if map.nside != self.nside:
                raise ValueError("Nside is not compatible")
            map = map.map

        if map.ndim == self.map.ndim:
            pass
        elif map.ndim == self.map.ndim + 1:
            self.map = self.map.reshape(self.map.shape[0:along_axis] + (1,) +
                                        self.map.shape[along_axis:])
        elif map.ndim == self.map.ndim - 1:
            map = map.reshape(map.shape[0:along_axis] + (1,) + 
                              map.shape[along_axis:])
        else:
            raise ValueError("Incompatible number of dimensions between maps")

        if along_axis == self.pixaxis:
            raise ValueError("Cannot append along pixel axis")
        if self.pol_axis is not None:
            if along_axis == self.pol_axis:
                raise ValueError("Cannot append along polarization axis")

        self.map = np.append(self.map, map, axis=along_axis)
