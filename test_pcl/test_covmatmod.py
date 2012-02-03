from __future__ import division
import covmatmod
import numpy as np
from nose.tools import ok_, eq_, assert_raises

nside = 32
npix = 12*nside**2
mat = shaperange(npix, npix, dtype=np.double)

def shaperange(shape, dtype=float):
    m = 1
    for s in shape:
        m = m * s
    srange = shaperange(m, dtype=dtype)
    srange = srange.reshape(shape)
    return srange.copy()

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
    yield ok_, np.all(md.mat == np.zeros(npix, npix))
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
    def func():
        md.ordering = 'neste'
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
    md.appendmats(mat, along_axis=4)
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
    mat = shaperange(npix, dtype=float)
    md = covmatmod.CovMatData(nside, mat=mat)
    for cmat in md:
        yield ok_, np.all(cmat == mat)
    mat = shaperange((3, npix))
    md = covmatmod.CovMatData(nside, mat=mat)
    currind = 0
    indlist = 3
    for cmat in md:
        yield ok_, np.all(mat[[currind,] + [Ellipsis,]] == cmat)
        yield ok_, cmat.shape == (npix,)
        currind += 1
    yield eq_, currind, 3
    mat  = shaperange(npix)
    md = covmatmod.CovMatData(nside, mat=mat)
    currind = 0
    for cmat in md:
        yield ok_, np.all(mat == cmat)
        currind += 1
    yield eq_, currind, 1
    #Keyword pol_iter=True should return (3, npix) or (npix, 3) - array for the
    #iterator
    mat = shaperange((2, 3, npix))
    md = covmatmod.CovMatData(nside, mat=mat, pol_axis=1, pol_iter=True)
    currind = 0
    for cmat in md:
        yield ok_, np.all(mat[[currind,] + [Ellipsis,]] == cmat)
        yield ok_, cmat.shape == (3, npix)
        currind += 1
    yield eq_, currind, 2
    mat = shaperange((5, npix, 3))
    md = covmatmod.CovMatData(nside, mat=mat, pol_axis=2, pol_iter=True)
    currind = 0
    for cmat in md:
        yield ok_, np.all(mat[[currind,] + [Ellipsis,]] == cmat)
        yield ok_, cmat.shape == (npix, 3)
        currind += 1
    yield eq_, currind, 5
    mat = shaperange((5, npix, 6, 3, 3))
    md = covmatmod.CovMatData(nside, mat=mat, pol_axis=3, pol_iter=True)
    currind = [0, 0, 0]
    indlist = [5, 6, 3]
    for cmat in md:
        yield ok_, np.all(mat[currind[:1] + [Ellipsis,] + currind[1:2] + 
                        [Ellipsis,] + currind[2:]] == cmat)
        yield ok_, cmat.shape == (npix, 3)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1
    mat = shaperange((4, 3, npix, 7, 1))
    md = covmatmod.CovMatData(nside, mat=mat, pol_axis=1, pol_iter=True)
    currind = [0, 0, 0]
    indlist = [4, 7, 1]
    for cmat in md:
        yield ok_, np.all(mat[currind[:1] + [Ellipsis,] + currind[1:]] == cmat)
        yield ok_, cmat.shape == (3, npix)
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
    yield ok_, np.all(md.mat * md2.mat == (md * md2).mat)
    yield ok_, np.all(md.mat - md2.mat == (md - md2).mat)
    md.mat = md.mat + 1
    md2.mat = md.mat + 1
    yield ok_, np.all(md.mat / md2.mat == (md / md2).mat)
    nmat = shaperange((npix, 5))
    md = covmatmod.CovMatData(nside=nside, mat=nmat)
    for i in range(5):
        yield ok_, np.all(md[:, i].mat == md.mat[:, i])
    yield eq_, md[:, 0].mat.shape, (npix,)
    #Different pixel axs in the resulting mat
    nmat = shaperange((3, npix))
    md = covmatmod.CovMatData(nside=nside, mat=nmat)
    for i in range(3):
        yield ok_, np.all(md[i].mat == md.mat[i])
    yield eq_, md[0].mat.shape, (npix,)
