from __future__ import division
import mapmod
import numpy as np
from nose.tools import ok_, eq_, assert_raises

nside = 32
npix = 12*nside**2
map = np.arange(npix)

def shaperange(shape, dtype=float):
    m = 1
    for s in shape:
        m = m * s
    srange = np.arange(m, dtype=dtype)
    srange = srange.reshape(shape)
    return srange.copy()

def test_ordering_conversion():
    r2npixs = {0:15, 44:95, 112:99, 55:52}
    n2rpixs = {0:74, 106:94, 66:135, 176:191}
    nside = 4
    npix = 12*nside**2
    map = np.arange(npix)
    md = mapmod.MapData(map=map, ordering='ring', nside=nside)
    md.switchordering()
    for key, value in r2npixs.items():
        yield eq_, md.map[key], value
    md.switchordering()
    yield ok_, np.all(map == md.map)

    md = mapmod.MapData(map=map, ordering='nested', nside=nside)
    md.switchordering()
    for key, value in n2rpixs.items():
        yield eq_, md.map[key], value
    md.switchordering()
    yield ok_, np.all(map == md.map)

    #Reload the module because init_r2n and init_n2r acts differently depending
    #on whether the other has been initialized:
    reload(mapmod)
    md = mapmod.MapData(map=map, ordering='nested', nside=nside)
    md.switchordering()
    for key, value in n2rpixs.items():
        yield eq_, md.map[key], value
    md.switchordering()
    yield ok_, np.all(map == md.map)

    md = mapmod.MapData(map=map, ordering='ring', nside=nside)
    md.switchordering()
    for key, value in r2npixs.items():
        yield eq_, md.map[key], value
    md.switchordering()
    yield ok_, np.all(map == md.map)

    #Test for other nsides
    nside = 2
    npix = 12*nside**2
    map = np.arange(npix)
    md = mapmod.MapData(map=map, ordering='ring', nside=nside)
    md.switchordering()
    md.switchordering()
    yield ok_, np.all(map == md.map)

    nside = 32
    npix = 12*nside**2
    map = np.arange(npix)
    md = mapmod.MapData(map=map, ordering='ring', nside=nside)
    md.switchordering()
    md.switchordering()
    yield ok_, np.all(map == md.map)

    #Test for other shapes
    map = shaperange((3, 4, npix))
    md = mapmod.MapData(nside, map=map)
    md.switchordering()
    md.switchordering()
    yield eq_, md.map.shape, (3, 4, npix)
    yield ok_, np.all(map == md.map)

    #Test the pixel versions as well
    nside = 4
    for key, value in r2npixs.items():
        yield eq_, mapmod.ring2nest_ind(key, nside), value
    for key, value in n2rpixs.items():
        yield eq_, mapmod.nest2ring_ind(key, nside), value


def test_sanity():
    md = mapmod.MapData(nside=nside)
    yield ok_, md.nside == nside
    md = mapmod.MapData(nside, ordering='ring')
    yield ok_, md.ordering == 'ring'
    md = mapmod.MapData(nside, ordering='nested')
    yield ok_, md.ordering == 'nested'
    md = mapmod.MapData(nside, map=map)
    yield ok_, np.all(md.map == map)

def test_init():
    map = np.arange(npix)
    def func():
        md = mapmod.MapData(nside=32.0)
    yield assert_raises, TypeError, func
    #Should not be able to set nside different from map nside
    def func():
        md = mapmod.MapData(nside=12, map=map)
    yield assert_raises, ValueError, func
    def func():
        md = mapmod.MapData(nside, ordering='ringe')
    yield assert_raises, ValueError, func
    def func():
        md = mapmod.MapData(nside, map=4)
    yield assert_raises, TypeError, func
    #Given no map, should initialize a map of zeros with given nside
    md = mapmod.MapData(nside=nside)
    yield ok_, np.all(md.map == np.zeros(npix))
    md = mapmod.MapData(nside, map=np.zeros((3, npix, 4)))
    yield eq_, (3, npix, 4), md.map.shape
    md = mapmod.MapData(nside, map=np.zeros((3, 7, npix)))
    yield eq_, (3, 7, npix), md.map.shape
    map = shaperange((3, npix, 5))
    def func():
        md = mapmod.MapData(nside, map=map, pix_axis=0)
    yield assert_raises, ValueError, func

def test_assign():
    md = mapmod.MapData(nside)
    def func():
        md.map = 4
    yield assert_raises, TypeError, func
    #Nside is immutable
    def func():
        md.nside = 12
    yield assert_raises, ValueError, func
    def func():
        md.ordering = 'neste'
    yield assert_raises, ValueError, func
    #Should be able to assign whichever map as long as nside is correct
    md = mapmod.MapData(nside)
    map = np.zeros((3, npix))
    try:
        md.map = map
    except:
        raise AssertionError()
    map = np.zeros((3, npix, 4))
    try:
        md.map = map
    except:
        raise AssertionError()
    map = np.zeros((1, 3, npix, 4, 5))
    try:
        md.map = map
    except:
        raise AssertionError()
    #Nside is immutable
    md = mapmod.MapData(nside)
    def func():
        md.nside = 2*nside
    yield assert_raises, ValueError, func

def test_shape():
    map = np.arange(npix)
    md = mapmod.MapData(nside)
    yield eq_, (npix,), md.map.shape
    md.map = np.zeros((4, npix, 5, 6))
    yield eq_, (4, npix, 5, 6), md.map.shape
    yield eq_, 1, md.pix_axis
    md.map = np.zeros((4, 5, 6, npix))
    yield eq_, (4, 5, 6, npix), md.map.shape
    yield eq_, 3, md.pix_axis
    map = np.resize(map, (3, npix, 5))
    md.map = map
    yield eq_, (3, npix, 5), md.map.shape
    yield eq_, 1, md.pix_axis

def test_pol():
    #Testing the polarization feature
    def func():
        md = mapmod.MapData(nside, pol_axis=0)
    yield assert_raises, ValueError, func
    map=np.zeros((3, npix))
    def func():
        md = mapmod.MapData(nside, map=map, pol_axis=1)
    yield assert_raises, ValueError, func
    try:
        md = mapmod.MapData(nside, map=map, pol_axis=0)
    except:
        raise AssertionError()
    md = mapmod.MapData(nside)
    yield eq_, md.pol_axis, None

def test_degrade():
    nside = 4
    npix = 12*nside**2
    map = np.arange(npix)
    nmap = np.arange(12*2*2)
    for i in range(12*2*2):
        sum = 0
        for j in range(4):
            sum += map[4*i + j]
        nmap[i] = sum
    nmap = nmap / 4
    md = mapmod.MapData(nside, map=map, ordering='nested')
    md = mapmod.degrade(md, nside_n=2)
    yield ok_, np.all(nmap == md.map)
    map = np.zeros((3, npix, 5))
    md = mapmod.MapData(nside, map=map)
    try:
        md = mapmod.degrade(md, nside_n=2)
    except:
        raise AssertionError()
    map = shaperange((npix, 1))
    nmap = nmap.reshape((12*2*2, 1))
    for i in range(3):
        map = np.append(map, map, axis=1)
        nmap = np.append(nmap, nmap, axis=1)
    md = mapmod.MapData(nside, map=map, ordering='nested')
    md = mapmod.degrade(md, nside_n=2)
    yield ok_, np.all(nmap == md.map)

def test_appendmaps():
    md = mapmod.MapData(nside)
    map = shaperange((1, npix))
    md.appendmaps(map, along_axis=0)
    combmap = np.append(np.zeros((1, npix)), map, axis=0)
    yield ok_, np.all(combmap == md.map)
    yield eq_, (2, npix), md.map.shape
    md = mapmod.MapData(nside, map=map)
    md.appendmaps(map, along_axis=0)
    combmap = np.append(map, map, axis=0)
    yield ok_, np.all(combmap == md.map)
    yield eq_, (2, npix), md.map.shape
    #Default axis should be 0:
    md = mapmod.MapData(nside, map=map)
    md.appendmaps(map)
    yield ok_, np.all(combmap == md.map)
    yield eq_, (2, npix), md.map.shape
    map = shaperange((3, 4, npix, 3, 1))
    md = mapmod.MapData(nside, map=map)
    def func():
        md.appendmaps(map, along_axis=2)
    yield assert_raises, ValueError, func
    md.appendmaps(map, along_axis=4)
    yield eq_, (3, 4, npix, 3, 2), md.map.shape

def test_iter():
    #Should iterate through the maps along pix_axis
    map = shaperange((3, 4, npix, 3))
    md = mapmod.MapData(nside, map=map)
    currind = [0, 0, 0]
    indlist = [3, 4, 3]
    for cmap in md:
        yield ok_, np.all(map[currind[:2] + [Ellipsis,] + currind[2:]] == cmap)
        yield ok_, cmap.shape == (npix,)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1
    map = np.arange(npix, dtype=float)
    md = mapmod.MapData(nside, map=map)
    for cmap in md:
        yield ok_, np.all(cmap == map)
    map = shaperange((3, npix))
    md = mapmod.MapData(nside, map=map)
    currind = 0
    indlist = 3
    for cmap in md:
        yield ok_, np.all(map[[currind,] + [Ellipsis,]] == cmap)
        yield ok_, cmap.shape == (npix,)
        currind += 1
    yield eq_, currind, 3
    map  = np.arange(npix)
    md = mapmod.MapData(nside, map=map)
    currind = 0
    for cmap in md:
        yield ok_, np.all(map == cmap)
        currind += 1
    yield eq_, currind, 1
    #Keyword pol_iter=True should return (3, npix) or (npix, 3) - array for the
    #iterator
    map = shaperange((2, 3, npix))
    md = mapmod.MapData(nside, map=map, pol_axis=1, pol_iter=True)
    currind = 0
    for cmap in md:
        yield ok_, np.all(map[[currind,] + [Ellipsis,]] == cmap)
        yield ok_, cmap.shape == (3, npix)
        currind += 1
    yield eq_, currind, 2
    map = shaperange((5, npix, 3))
    md = mapmod.MapData(nside, map=map, pol_axis=2, pol_iter=True)
    currind = 0
    for cmap in md:
        yield ok_, np.all(map[[currind,] + [Ellipsis,]] == cmap)
        yield ok_, cmap.shape == (npix, 3)
        currind += 1
    yield eq_, currind, 5
    map = shaperange((5, npix, 6, 3, 3))
    md = mapmod.MapData(nside, map=map, pol_axis=3, pol_iter=True)
    currind = [0, 0, 0]
    indlist = [5, 6, 3]
    for cmap in md:
        yield ok_, np.all(map[currind[:1] + [Ellipsis,] + currind[1:2] + 
                        [Ellipsis,] + currind[2:]] == cmap)
        yield ok_, cmap.shape == (npix, 3)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1
    map = shaperange((4, 3, npix, 7, 1))
    md = mapmod.MapData(nside, map=map, pol_axis=1, pol_iter=True)
    currind = [0, 0, 0]
    indlist = [4, 7, 1]
    for cmap in md:
        yield ok_, np.all(map[currind[:1] + [Ellipsis,] + currind[1:]] == cmap)
        yield ok_, cmap.shape == (3, npix)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1
