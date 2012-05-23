from __future__ import division
import numpy as np
import sys
from versionmod import bin, any, all
#import fileutils

_r2n = {}
_n2r = {}
_theta_r = {}
_phi_r = {}
_theta_n = {}
_phi_n = {}
_pix2x = None
_pix2y = None
_x2pix = None
_y2pix = None

def _init_pix2ang(nside, ordering):
#    pass
    global _theta_r
    global _phi_r
    global _theta_n
    global _phi_n
    global _x2pix
    global _y2pix
    
    npix = 12 * nside ** 2
#    if ordering == 'ring':
#        if _pix2ang_n.has_key(nside):
#            pass
#            #TODO ring2nest og saa tilbake
#    elif ordering == 'nested':
#        if _pix2ang_r.has_key(nside):
#            pass
#            #TODO omvendt
    pixs = np.arange(npix, dtype=int)
    theta = np.zeros(npix)
    phi = np.zeros(npix)
    if ordering == 'ring':
        nl2 = 2 * nside
        nl4 = 4 * nside
        ncap = nl2 * (nside - 1)
    
        #South polar cap
        filter = pixs >= npix - ncap
        ip = npix - pixs[filter]
        iring = (np.sqrt(ip * 0.5).round()).astype(int)
        iphi = 2 * iring * (iring + 1) - ip
        iring, iphi = _correct_ring_phi(-1, iring, iphi)
        theta[filter] = np.arccos((iring / nside) ** 2 / 3.0 - 1.0)
        phi[filter] = (iphi + 0.5) * np.pi * 0.5 / iring

        #Equatorial region
        filter = (pixs < npix - ncap) & (pixs >= ncap)
        ip = pixs[filter] - ncap
        iring = ip // nl4 + nside
        iphi = ip % nl4
        fodd = 0.5 * ((iring + nside + 1) % 2)
        theta[filter] = np.arccos((nl2 - iring) / (1.5 * nside))
        phi[filter] = (iphi + fodd) * np.pi * 0.5 / nside

        #North polar cap
        filter = pixs < ncap
        iring = np.sqrt(((pixs[filter] + 1) * 0.5).round()).astype(int)
        iphi = pixs[filter] - 2 * iring * (iring - 1)
        iring, iphi = _correct_ring_phi(1, iring, iphi)
        theta[filter] = np.arccos(1 - (iring / nside) ** 2 / 3)
        phi[filter] = (iphi + 0.5) * np.pi * 0.5 / iring

        _theta_r[nside] = theta
        _phi_r[nside] = phi
    elif ordering == 'nested':
        jrll = np.array((2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4))
        jpll = np.array((1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7))
        if _pix2x is None:
            _mk_pix2xy()
        npface = nside ** 2
        nl4 = 4 * nside

        face_num = (pixs / npface).astype(int)
        ipf = pixs % npface

        fn = nside
        fact1 = 1 / (3 * fn ** 2)
        fact2 = 2 / (3 * fn)
        ix = 0
        iy = 0
        scale = 1
        ismax = 4
        for i in range(0, ismax + 1):
            ip_low = ipf % 1024
            ix += scale * _pix2x[ip_low]
            iy += scale * _pix2y[ip_low]
            scale = scale * 32
            ipf = (ipf / 1024).astype(int)
        ix += scale * _pix2x[ipf]
        iy += scale * _pix2y[ipf]

        jrt = ix + iy
        jpt = ix - iy

        print face_num
        jr = jrll[face_num] * nside - jrt - 1

        kshift = np.zeros(len(pixs))
        nr = np.zeros(len(pixs))
        z = np.zeros(len(pixs))

        #South pole region
        filter = jr > 3 * nside
        nr[filter] = nl4 - jr[filter]
        z[filter] = - 1 + nr * fact1 * nr
        kshift[filter] = 0

        #Equatorial region
        filter = (jr <= 3 * nside) & (jr >= nside)
        nr[filter] = nside 
        z[filter] = (2 * nside - jr[filter]) * fact2
        kshift[filter] = (jr - nside) % 2

        #North pole region
        filter = jr < nside
        nr[filter] = jr[filter]
        z[filter] = 1 - nr[filter] * fact1 * nr[filter]
        kshift[filter] = 0

        theta = np.arccos(z)
        jp = (jpll[face_num] * nr + jpt + 1 + kshift) * 0.5
        jp[jp > nl4] -= nl4
        jp[jp < 1] -= nl4

        phi = (jp - (kshift + 1) * 0.5) * np.pi / nr
        _theta_n[nside] = theta
        _phi_n[nside] = phi

def _init_n2r(nside):
    global _r2n
    global _x2pix
    global _y2pix
    npix = 12 * nside ** 2
    #If the other is already initialized, use that information instead
    if _r2n.has_key(nside):
        _n2r[nside] = _r2n[nside].argsort()
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

    _n2r[nside] = ipf + face_num * nside ** 2

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

def _init_r2n(nside):
    global _r2n
    global _pix2x
    global _pix2y
    if _n2r.has_key(nside):
        _r2n[nside] = _n2r[nside].argsort()
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

    _r2n[nside] = n_before + jp - 1

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

def _compatible(md1, md2):
    if (md1.nside != md2.nside or md1.map.shape != md2.map.shape or 
        md1.pol_axis != md2.pol_axis or md1.pix_axis != md2.pix_axis or
        md1.ordering != md2.ordering or md1.pol_iter != md2.pol_iter):
        return False
    else:
        return True

def pix2ang(pix, ordering, nside):
    if ordering == 'ring':
        if not _theta_r.has_key(nside):
            _init_pix2ang(nside, ordering)
        return (_theta_r[nside][pix], _phi_r[nside][pix])
    elif ordering == 'nested':
        if not _theta_n.has_key(nside):
            _init_pix2ang(nside, ordering)
        return (_theta_n[nside][pix], _phi_n[nside][pix])


def degrade_average(md, nside_n):
    """Degrade input map to nside resolution by averaging over pixels.
    
    """
    switched = False
    if md.ordering == 'ring':
        switched = True
        md.switchordering()

    redfact = (md.nside // nside_n) ** 2
    temp = np.reshape(md.map, md.map.shape[:md.pix_axis] + 
                      (12*nside_n*nside_n, redfact) + 
                      md.map.shape[md.pix_axis + 1:])
    #nmap = MapData(nside_n, ordering='nested')
    md._nside = nside_n
    md.map = np.average(temp, axis=md.pix_axis + 1)
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
    md.map = md.map[(Ellipsis,) * md.pix_axis + (_r2n[md.nside],) + 
                    (Ellipsis,) * (md.map.ndim - 1 - md.pix_axis)]
    md.ordering='nested'

    return md

def nest2ring(md):
    global _n2r

    if not _n2r.has_key(md.nside):
        _init_n2r(md.nside)

    md.map = md.map[(Ellipsis,) * md.pix_axis + (_n2r[md.nside],) + 
                    (Ellipsis,) * (md.map.ndim - 1 - md.pix_axis)]
    md.ordering='ring'
    return md

def ring2nest_ind(ind, nside):
    global _n2r

    b = bin(nside)[2:]
    if (b[0] != '1' or int(b[1:], 2) != 0):
        raise ValueError('ring2nest_ind: nside has invalid value')
    if not _n2r.has_key(nside):
        _init_n2r(nside)

    return _n2r[nside][ind]

def nest2ring_ind(ind, nside):
    global _r2n

    b = bin(nside)[2:]
    if (b[0] != '1' or int(b[1:], 2) != 0):
        raise ValueError('nest2ring_ind: nside has invalid value')

    if not _r2n.has_key(nside):
        _init_r2n(nside)

    return _r2n[nside][ind]


class MapData(object):
    """Class to store and pass relevant HEALPix map information.

    The basic map structure is (nmaps, npix). Assignments with a
    one-dimensional array will reshape to this form. Ordering should be set
    at construction, subsequently it should be switched with switchordering().
    At this point, nside is uniquely determined by the last dimension of the
    map array, and all that is needed to initialize a map is an nside. This
    will make the MapData.map array be an array of zeroes. nside (and npix)
    for a map will be practically immutable after construction.
    Masks are also possible: if given a mask at initialization or later (with
    the setmask routine), the object will still contain all the original data, 
    but will also contain a mask2map array. The mask provided can be either a
    MapData object or numpy array with a boolean map array, 
    or a MapData object or numpy array with an integer map array, where the 
    masked pixels have the value 0, or a numpy array containing the pixel
    numbers to be masked. In the last case, the numpy array need not (and
    should not, unless you want to mask the whole map) contain as many elements
    as the original map. It is possible to specify different masks for
    polarization and temperature maps, but not for different maps, i.e. all 
    temperature maps in the MapData object must be masked the same way. See
    setmask for further documentation.


    """
    def __init__(self, nside, ordering='ring', map=None, pix_axis=None,
                 pol_axis=None, pol_iter=False, mask=None):
        if map is not None and pix_axis is not None:
            if map.shape[pix_axis] != 12 * nside ** 2:
                raise ValueError("""Explicit pix_axis does not contain the right
                                    number of pixels""")
        if pix_axis is None:
            pix_axis = 0
        self.pix_axis = pix_axis
        self._nside = None
        if map is None:
            if not isinstance(nside, int):
                raise TypeError("nside must be an integer")
            map = np.zeros(12*nside**2)
        self.nside = nside
        self.map = map
        self.ordering = ordering
        self.pol_axis = pol_axis
        self.pol_iter = pol_iter
        self.setmask(mask)

    def __iter__(self):
        return _map_iter(self)

    def __add__(self, other):
        if _compatible(self, other):
            return MapData(nside=self.nside, ordering=self.ordering, 
                           pix_axis=self.pix_axis, pol_axis=self.pol_axis,
                           pol_iter=self.pol_iter, map=self.map + other.map)
        else:
            raise ValueError("Maps not compatible for adding")

    def __mul__(self, other):
        if _compatible(self, other):
            return MapData(nside=self.nside, ordering=self.ordering, 
                           pix_axis=self.pix_axis, pol_axis=self.pol_axis,
                           pol_iter=self.pol_iter, map=self.map * other.map)
        else:
            raise ValueError("Maps not compatible for multiplying")

    def __sub__(self, other):
        if _compatible(self, other):
            return MapData(nside=self.nside, ordering=self.ordering, 
                           pix_axis=self.pix_axis, pol_axis=self.pol_axis,
                           pol_iter=self.pol_iter, map=self.map - other.map)
        else:
            raise ValueError("Maps not compatible for subtracting")

    def __truediv__(self, other):
        if _compatible(self, other):
            return MapData(nside=self.nside, ordering=self.ordering, 
                           pix_axis=self.pix_axis, pol_axis=self.pol_axis,
                           pol_iter=self.pol_iter, map=self.map / other.map)
        else:
            raise ValueError("Maps not compatible for dividing")

    def __getitem__(self, index):
        n = MapData(nside=self.nside, ordering=self.ordering, 
                       pol_iter=self.pol_iter, map=self.map[index])
        if n.pix_axis == self.pix_axis or self.pol_axis is None:
            n.pol_axis = self.pol_axis
        else:
            n.pol_axis = None
        return n

    def getmap(self):
        return self._map

    def setmap(self, map):
        if not isinstance(map, np.ndarray):
            raise TypeError("Map must be numpy array")
        if (self.pix_axis >= map.ndim or 
                map.shape[self.pix_axis] != 12*self.nside**2):
            #Try to autodetect pixel axis
            for i in range(map.ndim):
                if map.shape[i] == 12*self.nside**2:
                    self.pix_axis = i
                    break
            else:
                raise ValueError("""Pixel number of input map does not conform 
                                    to nside""")
        self._map = map

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

        if along_axis == self.pix_axis:
            raise ValueError("Cannot append along pixel axis")
        if self.pol_axis is not None:
            if along_axis == self.pol_axis:
                raise ValueError("Cannot append along polarization axis")

        self.map = np.append(self.map, map, axis=along_axis)

    def setmask(self, mask):
        """Routine to set the mask of the MapData object.

        Keyword arguments:
        mask -- None, MapData object or numpy.ndarray object.

        This routine is called at initialization. If mask is None, the 'masked'
        attribute will be set to False, and mask and mask2map will be set to
        None. If mask is a numpy array either contained in a MapData object or
        just by itself, it will set self.mask2map and self.mask appropriately. 
        self.mask will be a logical array, self.mask2map an integer array. The
        mask array can contain either zeroes and ones, booleans, or a list of 
        pixel indices to be masked. Different masks for different polarizations
        are supported, the relevant arrays then must have shape (3, npix) or
        (npix, 3), or (number of masked pixels, 3) or 
        (3, number of masked pixels) in the case where the array provided
        contains the actual number of pixels to be masked. 
        There is potential for confusion here, so to be sure that the mask 
        array is interpreted correctly when it contains integer values, *only* 
        use ones and zeros if mask is the actual mask - do not use values 
        greater than one. Also, in the very unlikely event that there are 
        three pixels to be masked and this is specified by using the actual
        pixel values, the routine will have trouble identifying the
        axis that represents polarization.

        """
        npix = 12 * self.nside ** 2
        if self.pol_axis is not None:
            ndim = 3
        else:
            ndim = 1
        if mask is None:
            self.mask2map = None
            self.mask = None
            self.masked = False
            self.npix_masked = np.array(ndim * [npix])
            return
        elif isinstance(mask, MapData):
            if mask.nside != self.nside:
                raise ValueError("Mask nside and map nside are incompatible")
            if mask.masked:
                raise ValueError("Mask is itself masked. Absurd")
            if self.pol_axis is None and mask.pol_axis is not None:
                raise ValueError("""Mask is polarised but map is not 
                                 (or at least it is ambiguous whether it is 
                                 or not)""")
            if len(mask.map.shape) > 2:
                raise ValueError("""Mask cannot contain more than 2 
                                    dimensions""")
            maskpol = mask.pol_axis
            tmask = mask.map
        elif isinstance(mask, np.ndarray):
            if len(mask.shape) == 2:
                if (mask.shape == (3, 3) or mask.shape == (1, 3) or 
                        mask.shape == (3, 1)):
                    raise ValueError("""Please use the other mask type because
                                        this is ambiguous""")
                elif (mask.shape[0] == 3):
                    maskpol = 0
                elif (mask.shape[1] == 3):
                    maskpol = 1
                else:
                    if not (mask.shape[0] == 1 or mask.shape[1] == 1):
                        raise ValueError("""Mask contains two dimensions but 
                                            they're both neither one nor 
                                            three""")
                    else:
                        maskpol = None
            elif len(mask.shape) == 1:
                maskpol = None
            else: 
                raise ValueError("""Mask cannot contain more than 2
                                    dimensions or less than 1 dimension""")
            if ((np.size(mask) != npix and np.size(mask) != 3 * npix) or 
                    np.any(mask > 1)):
                #Assume it contains pixel indices - this only happens for the
                #ndarray case.
                if mask.dtype != 'int':
                    raise TypeError("""Mask supposed to specify pixel values
                                        but does not have integer data type""")
                if len(mask.shape) == 1:
                    if any((np.sort(mask)[1:] - np.sort(mask)[:-1]) == 0):
                        raise ValueError("""Duplicate values for mask that 
                                            specifies which pixels to mask""")
                else:
                    if maskpol is None:
                        if any(np.sort(np.reshape(mask, np.size(mask)))[1:] - 
                                np.sort(np.reshape(mask, np.size(mask)))[:-1] 
                                == 0):
                            raise ValueError("""Duplicate values for mask that 
                                                specifies which pixels to 
                                                mask""")
                    elif maskpol == 0:
                        for i in range(3):
                            if any(np.sort(mask[i])[1:] - np.sort(mask[i])[:-1]
                                    == 0):
                                raise ValueError("""Duplicate values for mask 
                                                    that specifies which pixels
                                                    to mask""")
                    elif maskpol == 1:
                        for i in range(3):
                            if any(np.sort(mask[:, i])[1:] - 
                                    np.sort(mask[:, i])[:-1] == 0):
                                raise ValueError("""Duplicate values for mask 
                                                    that specifies which pixels
                                                    to mask""")
                if len(mask.shape) == 1:
                    tmask = np.ones(npix)
                    tmask[mask] = 0
                elif len(mask.shape) == 2:
                    if maskpol is None:
                        tmask = np.ones(npix)
                        tmask[np.reshape(mask, np.size(mask))] = 0
                    elif maskpol == 0:
                        tmask = np.ones((3, npix))
                        for i in range(3):
                            tmask[i, mask[i]] = 0
                    elif maskpol == 1:
                        tmask = np.ones((npix, 3))
                        for i in range(3):
                            tmask[mask[i], i] = 0
            else:
                tmask = mask
        else:
            raise TypeError("""Wrong object for mask""")

        self.mask = np.zeros((ndim, 12 * self.nside ** 2), dtype='bool')
        if (tmask.dtype == 'int' or tmask.dtype == 'float' or 
                tmask.dtype == 'bool'):
            if maskpol is None:
                if len(tmask.shape) == 1:
                    self.mask[:, :] = ndim * [tmask != 0]
                else:
                    self.mask[:, :] = (ndim * 
                            [np.reshape(tmask, np.size(tmask)) != 0])
            elif maskpol == 0:
                self.mask = (tmask != 0)
            elif maskpol == 1:
                self.mask = (np.transpose(tmask) != 0)
        else:
            raise TypeError("""Mask datatype is not supported""")

        #Make the mask2map array
        self.npix_masked = np.zeros(ndim)
        allpixs = np.arange(npix, dtype='int')
        self.mask2map = np.zeros((ndim, npix), dtype='int')

        for i in range(ndim):
            self.npix_masked[i] = sum(self.mask[i])
            self.mask2map[i, :self.npix_masked[i]] = allpixs[self.mask[i]]
        self.masked = True

class _map_iter(object):
    def __init__(self, md):
        if not isinstance(md, MapData):
            raise TypeError()
        self._currmap = 1
        if md.pol_iter == True and md.pol_axis is None:
            raise ValueError("pol_iter is True but no pol_axis given")
        self._pol_iter = md.pol_iter
        if self._pol_iter:
            if md.pix_axis < md.pol_axis:
                self._subshape = list(md.map.shape[:md.pix_axis] + 
                                    md.map.shape[md.pix_axis + 1:md.pol_axis] +
                                    md.map.shape[md.pol_axis + 1:])
            else:
                self._subshape = list(md.map.shape[:md.pol_axis] + 
                                    md.map.shape[md.pol_axis + 1:md.pix_axis] +
                                    md.map.shape[md.pix_axis + 1:])
        else:
            self._subshape = list(md.map.shape[:md.pix_axis] + 
                            md.map.shape[md.pix_axis + 1:])
        for dim in self._subshape:
            self._currmap *= dim
        self._map = md.map
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
        if self._currmap == 0:
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
        self._currmap -= 1
        if self._pol_iter:
            if self._pix_axis < self._pol_axis:
                return self._map[self._currind[:self._pix_axis] + [Ellipsis,]
                            + self._currind[self._pix_axis:self._pol_axis] 
                            + [Ellipsis,] + self._currind[self._pol_axis:]]
            else:
                return self._map[self._currind[:self._pol_axis] + [Ellipsis,]
                            + self._currind[self._pol_axis:self._pix_axis] 
                            + [Ellipsis,] + self._currind[self._pix_axis:]]
        else:
            return self._map[self._currind[:self._pix_axis] + [Ellipsis,] 
                        + self._currind[self._pix_axis:]]
