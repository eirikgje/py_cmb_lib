from __future__ import division
import covmatmod
import mapmod
import numpy as np
from nose.tools import ok_, eq_, assert_raises


def shaperange(shape, dtype=float):
    m = 1
    for s in shape:
        m = m * s
    srange = np.arange(m, dtype=dtype)
    srange = srange.reshape(shape)
    return srange.copy()

nside = 8
npix = 12*nside**2
mat = shaperange((npix, npix), dtype=np.double)

def test_sanity():
    md = covmatmod.CovMatData(nside=nside)
    yield ok_, md.nside == nside
    md = covmatmod.CovMatData(nside, ordering='ring')
    yield ok_, md.ordering == 'ring'
    md = covmatmod.CovMatData(nside, ordering='nested')
    yield ok_, md.ordering == 'nested'
    md = covmatmod.CovMatData(nside, mat=mat)
    yield ok_, np.all(md.mat == mat)

def test_init():
    mat = shaperange((npix, npix))
    def func():
        md = covmatmod.CovMatData(nside=32.0)
    yield assert_raises, TypeError, func
    #Should not be able to set nside different from mat nside
    def func():
        md = covmatmod.CovMatData(nside=12, mat=mat)
    yield assert_raises, ValueError, func
    def func():
        md = covmatmod.CovMatData(nside, ordering='ringe')
    yield assert_raises, ValueError, func
    def func():
        md = covmatmod.CovMatData(nside, mat=4)
    yield assert_raises, TypeError, func
    #Given no mat, should initialize a mat of zeros with given nside
    md = covmatmod.CovMatData(nside=nside)
    yield ok_, np.all(md.mat == np.zeros((npix, npix)))
    md = covmatmod.CovMatData(nside, mat=np.zeros((3, npix, npix, 4)))
    yield eq_, (3, npix, npix, 4), md.mat.shape
    md = covmatmod.CovMatData(nside, mat=np.zeros((3, 7, npix, npix)))
    yield eq_, (3, 7, npix, npix), md.mat.shape
    mat = shaperange((3, npix, npix, 5))
    def func():
        md = covmatmod.CovMatData(nside, mat=mat, pix_axis=0)
    yield assert_raises, ValueError, func

def test_assign():
    md = covmatmod.CovMatData(nside)
    def func():
        md.mat = 4
    yield assert_raises, TypeError, func
    #Nside is immutable
    def func():
        md.nside = 12
    yield assert_raises, ValueError, func
    #Ordering is immutable
    md = covmatmod.CovMatData(nside, ordering='ring')
    def func():
        md.ordering = 'nested'
    yield assert_raises, ValueError, func
    #Should be able to assign whichever mat as long as nside is correct
    md = covmatmod.CovMatData(nside)
    mat = np.zeros((3, npix, npix))
    try:
        md.mat = mat
    except:
        raise AssertionError()
    mat = np.zeros((3, npix, npix, 4))
    try:
        md.mat = mat
    except:
        raise AssertionError()
    mat = np.zeros((1, 3, npix, npix, 4, 5))
    try:
        md.mat = mat
    except:
        raise AssertionError()
    #Nside is immutable
    md = covmatmod.CovMatData(nside)
    def func():
        md.nside = 2*nside
    yield assert_raises, ValueError, func

def test_shape():
    md = covmatmod.CovMatData(nside)
    yield eq_, (npix, npix), md.mat.shape
    md.mat = np.zeros((4, npix, npix, 5, 6))
    yield eq_, (4, npix, npix, 5, 6), md.mat.shape
    yield eq_, 1, md.pix_axis
    md.mat = np.zeros((4, 5, 6, npix, npix))
    yield eq_, (4, 5, 6, npix, npix), md.mat.shape
    yield eq_, 3, md.pix_axis
    mat = shaperange((npix, npix))
    mat = np.resize(mat, (3, npix, npix, 5))
    md.mat = mat
    yield eq_, (3, npix, npix, 5), md.mat.shape
    yield eq_, 1, md.pix_axis

def test_pol():
    #Testing the polarization feature
    def func():
        md = covmatmod.CovMatData(nside, pol_axis=0)
    yield assert_raises, ValueError, func
    mat=np.zeros((3, npix, npix))
    def func():
        md = covmatmod.CovMatData(nside, mat=mat, pol_axis=1)
    yield assert_raises, ValueError, func
    try:
        md = covmatmod.CovMatData(nside, mat=mat, pol_axis=0)
    except:
        raise AssertionError()
    md = covmatmod.CovMatData(nside)
    yield eq_, md.pol_axis, None

def test_appendmats():
    md = covmatmod.CovMatData(nside)
    mat = shaperange((1, npix, npix))
    md.appendmats(mat, along_axis=0)
    combmat = np.append(np.zeros((1, npix, npix)), mat, axis=0)
    yield ok_, np.all(combmat == md.mat)
    yield eq_, (2, npix, npix), md.mat.shape
    md = covmatmod.CovMatData(nside, mat=mat)
    md.appendmats(mat, along_axis=0)
    combmat = np.append(mat, mat, axis=0)
    yield ok_, np.all(combmat == md.mat)
    yield eq_, (2, npix, npix), md.mat.shape
    #Default axis should be 0:
    md = covmatmod.CovMatData(nside, mat=mat)
    md.appendmats(mat)
    yield ok_, np.all(combmat == md.mat)
    yield eq_, (2, npix, npix), md.mat.shape
    mat = shaperange((3, 4, npix, npix, 3, 1))
    md = covmatmod.CovMatData(nside, mat=mat)
    def func():
        md.appendmats(mat, along_axis=2)
    yield assert_raises, ValueError, func
    md.appendmats(mat, along_axis=5)
    yield eq_, (3, 4, npix, npix, 3, 2), md.mat.shape

def test_iter():
    #Should iterate through the mats along pix_axis
    mat = shaperange((3, 4, npix, npix, 3))
    md = covmatmod.CovMatData(nside, mat=mat)
    currind = [0, 0, 0]
    indlist = [3, 4, 3]
    for cmat in md:
        yield ok_, np.all(mat[currind[:2] + [Ellipsis,] + currind[2:]] == cmat)
        yield ok_, cmat.shape == (npix, npix)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1
    mat = shaperange((npix, npix), dtype=float)
    md = covmatmod.CovMatData(nside, mat=mat)
    for cmat in md:
        yield ok_, np.all(cmat == mat)
    mat = shaperange((3, npix, npix))
    md = covmatmod.CovMatData(nside, mat=mat)
    currind = 0
    indlist = 3
    for cmat in md:
        yield ok_, np.all(mat[[currind,] + [Ellipsis,]] == cmat)
        yield ok_, cmat.shape == (npix, npix)
        currind += 1
    yield eq_, currind, 3
    mat  = shaperange((npix, npix))
    md = covmatmod.CovMatData(nside, mat=mat)
    currind = 0
    for cmat in md:
        yield ok_, np.all(mat == cmat)
        currind += 1
    yield eq_, currind, 1
    #Keyword pol_iter=True should return (3, npix) or (npix, 3) - array for the
    #iterator
    mat = shaperange((2, 3, npix, npix))
    md = covmatmod.CovMatData(nside, mat=mat, pol_axis=1, pol_iter=True)
    currind = 0
    for cmat in md:
        yield ok_, np.all(mat[[currind,] + [Ellipsis,]] == cmat)
        yield ok_, cmat.shape == (3, npix, npix)
        currind += 1
    yield eq_, currind, 2
    mat = shaperange((5, npix, npix, 3))
    md = covmatmod.CovMatData(nside, mat=mat, pol_axis=3, pol_iter=True)
    currind = 0
    for cmat in md:
        yield ok_, np.all(mat[[currind,] + [Ellipsis,]] == cmat)
        yield ok_, cmat.shape == (npix, npix, 3)
        currind += 1
    yield eq_, currind, 5
    mat = shaperange((5, npix, npix, 6, 3, 3))
    md = covmatmod.CovMatData(nside, mat=mat, pol_axis=4, pol_iter=True)
    currind = [0, 0, 0]
    indlist = [5, 6, 3]
    for cmat in md:
        yield ok_, np.all(mat[currind[:1] + [Ellipsis,] + currind[1:2] + 
                [Ellipsis,] + currind[2:]] == cmat)
        yield ok_, cmat.shape == (npix, npix, 3)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1
    mat = shaperange((4, 3, npix, npix, 7, 1))
    md = covmatmod.CovMatData(nside, mat=mat, pol_axis=1, pol_iter=True)
    currind = [0, 0, 0]
    indlist = [4, 7, 1]
    for cmat in md:
        yield ok_, np.all(mat[currind[:1] + [Ellipsis,] + currind[1:]] == cmat)
        yield ok_, cmat.shape == (3, npix, npix)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1

    mat = shaperange((4, 7, npix, npix, 3, 1))
    md = covmatmod.CovMatData(nside, mat=mat, pol_axis=4, pol_iter=True)
    currind = [0, 0, 0]
    indlist = [4, 7, 1]
    for cmat in md:
        yield ok_, np.all(mat[currind[:2] + [Ellipsis,] + currind[2:]] == cmat)
        yield ok_, cmat.shape == (npix, npix, 3)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1


def test_operators():
    md = covmatmod.CovMatData(nside=nside, mat=mat)
    md2 = covmatmod.CovMatData(nside=nside, mat=mat)
    yield ok_, np.all(md.mat + md2.mat == (md + md2).mat)
    yield ok_, np.all(md.mat + md.mat == (md + md).mat)
    #Multiplication not implemented yet
    #yield ok_, np.all(md.mat * md2.mat == (md * md2).mat)
    yield ok_, np.all(md.mat - md2.mat == (md - md2).mat)
    #Division not implemented yet
    #md.mat = md.mat + 1
    #md2.mat = md.mat + 1
    #yield ok_, np.all(md.mat / md2.mat == (md / md2).mat)

    #Getitem must be reimplemented to return a slice, not a copy
    #nmat = shaperange((npix, npix, 5))
    #md = covmatmod.CovMatData(nside=nside, mat=nmat)
    #for i in range(5):
    #    yield ok_, np.all(md[:, i].mat == md.mat[:, i])
    #yield eq_, md[:, 0].mat.shape, (npix, npix)
    #Different pixel axs in the resulting matrix
    #nmat = shaperange((3, npix, npix))
    #md = covmatmod.CovMatData(nside=nside, mat=nmat)
    #for i in range(3):
    #    yield ok_, np.all(md[i].mat == md.mat[i])
    #yield eq_, md[0].mat.shape, (npix, npix)

def test_mask():
    #Mask is ones and zeros
    mat = shaperange((npix, npix))
    mask = np.ones(npix) 
    mask[5] = 0
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=mask)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = covmatmod.CovMatData(nside=nside, mat=mat)
    mask2map = np.arange(npix)
    #yield ok_, np.all(mask2map == md.mask2map)
    yield ok_, md.mask2map is None
    md.setmask(mask)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Let mask be matData object
    maskd = mapmod.MapData(nside=nside, map=mask)
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=maskd)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = covmatmod.CovMatData(nside=nside, mat=mat)
    md.setmask(maskd)
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is boolean array
    mask = np.zeros(npix, dtype=bool)
    mask[:] = True
    mask[8] = False
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=mask)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:8],  mask2map[9:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is CovMatData object
    maskd = mapmod.MapData(nside=nside, map=mask)
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=maskd)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:8],  mask2map[9:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = covmatmod.CovMatData(nside=nside, mat=mat)
    md.setmask(maskd)
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is numpy array containing the pixels to be masked
    mask = np.array([3, 6, 19, 54, 100])
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=mask)
    mask2map=np.arange(npix)
    mask2map = np.concatenate((mask2map[:3],  mask2map[4:6], mask2map[7:19],
                                mask2map[20:54], mask2map[55:100],
                                mask2map[101:]))
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Test multiple mats
    mat = shaperange((4, npix, npix, 2))

    #Mask is ones and zeros
    mask = np.ones(npix) 
    mask[5] = 0
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=mask)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = covmatmod.CovMatData(nside=nside, mat=mat)
    mask2map = np.arange(npix)
    #yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, md.mask2map is None
    md.setmask(mask)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Let mask be CovMatData object
    maskd = mapmod.MapData(nside=nside, map=mask)
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=maskd)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:5],  mask2map[6:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = covmatmod.CovMatData(nside=nside, mat=mat)
    md.setmask(maskd)
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is boolean array
    mask = np.zeros(npix, dtype=bool)
    mask[:] = True
    mask[8] = False
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=mask)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:8],  mask2map[9:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is CovMatData object
    maskd = mapmod.MapData(nside=nside, map=mask)
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=maskd)
    mask2map = np.arange(npix)
    mask2map = np.append(mask2map[:8],  mask2map[9:])
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    md = covmatmod.CovMatData(nside=nside, mat=mat)
    md.setmask(maskd)
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Mask is numpy array containing the pixels to be masked
    mask = np.array([3, 6, 19, 54, 100])
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=mask)
    mask2map=np.arange(npix)
    mask2map = np.concatenate((mask2map[:3], mask2map[4:6], mask2map[7:19], 
                        mask2map[20:54], mask2map[55:100], 
                        mask2map[101:]))
    yield ok_, np.all(mask2map == md.mask2map[0, :md.npix_masked[0]])

    #Test different masks for polarization
    mat = shaperange((3, npix, npix))

    #Mask is ones and zeros
    mask = np.ones((3, npix)) 
    mask[0, 5] = 0
    mask[1, [6, 9, 15]] = 0
    mask[2, [7, 9, 15, 90]] = 0
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=mask, pol_axis=0)
    mask2map = np.arange(npix)
    mask2mapT = np.append(mask2map[:5], mask2map[6:])
    mask2mapQ = np.concatenate((mask2map[:6], mask2map[7:9], mask2map[10:15], 
                            mask2map[16:]))
    mask2mapU = np.concatenate((mask2map[:7], mask2map[8:9], mask2map[10:15], 
                            mask2map[16:90], mask2map[91:]))
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    md = covmatmod.CovMatData(nside=nside, mat=mat, pol_axis=0)
    #yield ok_, np.all(mask2map[0] == md.mask2map[0])
    #yield ok_, np.all(mask2map[1] == md.mask2map[1])
    #yield ok_, np.all(mask2map[2] == md.mask2map[2])
    yield ok_, md.mask2map is None
    md.setmask(mask)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    #Let mask be CovMatData object
    maskd = mapmod.MapData(nside=nside, map=mask, pol_axis=0)
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=maskd, pol_axis=0)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    md = covmatmod.CovMatData(nside=nside, mat=mat, pol_axis=0)
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
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=mask, pol_axis=0)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    #Mask is CovMatData object
    maskd = mapmod.MapData(nside=nside, map=mask, pol_axis=0)
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=maskd, pol_axis=0)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    md = covmatmod.CovMatData(nside=nside, mat=mat, pol_axis=0)
    md.setmask(maskd)
    yield ok_, np.all(mask2mapT == md.mask2map[0, :md.npix_masked[0]])
    yield ok_, np.all(mask2mapQ == md.mask2map[1, :md.npix_masked[1]])
    yield ok_, np.all(mask2mapU == md.mask2map[2, :md.npix_masked[2]])

    #Mask is numpy array containing the pixels to be masked. The array must then
    #contain the same number of pixels for polarization and temperature
    mask = np.array([[3, 6, 19, 54, 100], [5, 8, 101, 302, 689], 
                    [1, 5, 100, 250, 600]])
    md = covmatmod.CovMatData(nside=nside, mat=mat, mask=mask, pol_axis=0)
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
