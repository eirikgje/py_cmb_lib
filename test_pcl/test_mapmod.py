from __future__ import division
import mapmod
import numpy as np
from nose.tools import ok_, eq_, assert_raises

nside = 32
npix = 12*nside**2
map = np.arange(npix, dtype=np.double)

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

    nside = 8
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

def test_operators():
    md = mapmod.MapData(nside=nside, map=map)
    md2 = mapmod.MapData(nside=nside, map=map)
    yield ok_, np.all(md.map + md2.map == (md + md2).map)
    yield ok_, np.all(md.map + md.map == (md + md).map)
    yield ok_, np.all(md.map * md2.map == (md * md2).map)
    yield ok_, np.all(md.map - md2.map == (md - md2).map)
    md.map = md.map + 1
    md2.map = md.map + 1
    yield ok_, np.all(md.map / md2.map == (md / md2).map)
    #Getitem must be reimplemented to return a slice, not a copy
    #nmap = shaperange((npix, 5))
    #md = mapmod.MapData(nside=nside, map=nmap)
    #for i in range(5):
    #    yield ok_, np.all(md[:, i].map == md.map[:, i])
    #yield eq_, md[:, 0].map.shape, (npix,)
    #Different pixel axs in the resulting map
    #nmap = shaperange((3, npix))
    #md = mapmod.MapData(nside=nside, map=nmap)
    #for i in range(3):
    #    yield ok_, np.all(md[i].map == md.map[i])
    #yield eq_, md[0].map.shape, (npix,)

def test_mask():
    #Mask is ones and zeros
    map = np.arange(npix)
    mask = np.ones(npix) 
    mask[5] = 0
    md = mapmod.MapData(nside=nside, map=map, mask=mask)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = mapmod.MapData(nside=nside, map=map)
    mask2map = np.arange(npix)
    #yield ok_, np.all(mask2map == md.mask2map)
    yield ok_, md.mask2map is None
    md.setmask(mask)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Let mask be mapdata object
    maskd = mapmod.MapData(nside=nside, map=mask)
    md = mapmod.MapData(nside=nside, map=map, mask=maskd)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = mapmod.MapData(nside=nside, map=map)
    md.setmask(maskd)
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is boolean array
    mask = np.zeros(npix, dtype=bool)
    mask[:] = True
    mask[8] = False
    md = mapmod.MapData(nside=nside, map=map, mask=mask)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:8],  mask2map[9:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is mapdata object
    maskd = mapmod.MapData(nside=nside, map=mask)
    md = mapmod.MapData(nside=nside, map=map, mask=maskd)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:8],  mask2map[9:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = mapmod.MapData(nside=nside, map=map)
    md.setmask(maskd)
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is numpy array containing the pixels to be masked
    mask = np.array([3, 6, 19, 54, 100])
    md = mapmod.MapData(nside=nside, map=map, mask=mask)
    mask2map=np.arange(npix)
    mask2map = np.concatenate((mask2map[:3],  mask2map[4:6], mask2map[7:19],
                                mask2map[20:54], mask2map[55:100],
                                mask2map[101:]))
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Test multiple maps
    map = shaperange((4, npix, 2))

    #Mask is ones and zeros
    mask = np.ones(npix) 
    mask[5] = 0
    md = mapmod.MapData(nside=nside, map=map, mask=mask)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = mapmod.MapData(nside=nside, map=map)
    mask2map = np.arange(npix)
    #yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, md.mask2map is None
    md.setmask(mask)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Let mask be mapdata object
    maskd = mapmod.MapData(nside=nside, map=mask)
    md = mapmod.MapData(nside=nside, map=map, mask=maskd)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = mapmod.MapData(nside=nside, map=map)
    md.setmask(maskd)
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is boolean array
    mask = np.zeros(npix, dtype=bool)
    mask[:] = True
    mask[8] = False
    md = mapmod.MapData(nside=nside, map=map, mask=mask)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:8],  mask2map[9:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is mapdata object
    maskd = mapmod.MapData(nside=nside, map=mask)
    md = mapmod.MapData(nside=nside, map=map, mask=maskd)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:8],  mask2map[9:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = mapmod.MapData(nside=nside, map=map)
    md.setmask(maskd)
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is numpy array containing the pixels to be masked
    mask = np.array([3, 6, 19, 54, 100])
    md = mapmod.MapData(nside=nside, map=map, mask=mask)
    mask2map=np.arange(npix)
    mask2map = np.concatenate((mask2map[:3], mask2map[4:6], mask2map[7:19], 
                        mask2map[20:54], mask2map[55:100], 
                        mask2map[101:]))
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Test different masks for polarization
    map = shaperange((3, npix))

    #Mask is ones and zeros
    mask = np.ones((3, npix)) 
    mask[0, 5] = 0
    mask[1, [6, 9, 15]] = 0
    mask[2, [7, 9, 15, 90]] = 0
    md = mapmod.MapData(nside=nside, map=map, mask=mask, pol_axis=0)
    mask2map = np.arange(npix)
    mask2mapT = np.append(mask2map[:5], mask2map[6:])
    mask2mapQ = np.concatenate((mask2map[:6], mask2map[7:9], mask2map[10:15], 
                            mask2map[16:]))
    mask2mapU = np.concatenate((mask2map[:7], mask2map[8:9], mask2map[10:15], 
                            mask2map[16:90], mask2map[91:]))
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    md = mapmod.MapData(nside=nside, map=map, pol_axis=0)
    #yield ok_, np.all(mask2map[0] == md.mask2map[0])
    #yield ok_, np.all(mask2map[1] == md.mask2map[1])
    #yield ok_, np.all(mask2map[2] == md.mask2map[2])
    yield ok_, md.mask2map is None
    md.setmask(mask)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    #Let mask be mapdata object
    maskd = mapmod.MapData(nside=nside, map=mask, pol_axis=0)
    md = mapmod.MapData(nside=nside, map=map, mask=maskd, pol_axis=0)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    md = mapmod.MapData(nside=nside, map=map, pol_axis=0)
    md.setmask(maskd)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    #Mask is boolean array
    mask = np.zeros((3, npix), dtype=bool) 
    mask[:] = True
    mask[0, 5] = False
    mask[1, [6, 9, 15]] = False
    mask[2, [7, 9, 15, 90]] = False
    md = mapmod.MapData(nside=nside, map=map, mask=mask, pol_axis=0)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    #Mask is mapdata object
    maskd = mapmod.MapData(nside=nside, map=mask, pol_axis=0)
    md = mapmod.MapData(nside=nside, map=map, mask=maskd, pol_axis=0)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    md = mapmod.MapData(nside=nside, map=map, pol_axis=0)
    md.setmask(maskd)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    #Mask is numpy array containing the pixels to be masked. The array must then
    #contain the same number of pixels for polarization and temperature
    mask = np.array([[3, 6, 19, 54, 100], [5, 8, 101, 302, 689], 
                    [1, 5, 100, 250, 600]])
    md = mapmod.MapData(nside=nside, map=map, mask=mask, pol_axis=0)
    mask2map=np.arange(npix)
    mask2mapT = np.concatenate((mask2map[:3], mask2map[4:6], mask2map[7:19], 
                                mask2map[20:54], mask2map[55:100], 
                                mask2map[101:]))
    mask2mapQ = np.concatenate((mask2map[:5], mask2map[6:8], mask2map[9:101], 
                                mask2map[102:302], mask2map[303:689], 
                                mask2map[690:]))
    mask2mapU = np.concatenate((mask2map[:1], mask2map[2:5], mask2map[6:100], 
                                mask2map[101:250], mask2map[251:600], 
                                mask2map[601:]))
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])
