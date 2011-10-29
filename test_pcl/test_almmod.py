from __future__ import division
import almmod
import numpy as np
from nose.tools import ok_, eq_, assert_raises
import sys


def shaperange(shape, dtype=float):
    m = 1
    for s in shape:
        m = m * s
    srange = np.arange(m, dtype=dtype)
    srange = srange.reshape(shape)
    return srange.copy()

def shaperange_cplx(shape):
    m = 1
    for s in shape:
        m = m * s
    out = np.zeros(m, dtype = np.complex)
    for i in range(int(m)):
        out[i] = np.complex(i, m + i)
    out = out.reshape(shape)
    return out.copy()

lmax = 95
nels = lmax * (lmax + 1) // 2 + lmax + 1
alms = shaperange_cplx((nels,))
cls = shaperange((lmax + 1,), dtype=float)

def test_sanity():
    alms = shaperange_cplx((nels,))
    ad = almmod.AlmData(lmax)
    yield ok_, ad.lmax == lmax
    yield ok_, ad.alms.dtype == np.complex
    ad = almmod.AlmData(lmax, alms=alms)
    yield ok_, np.all(ad.alms == alms)
    alms = shaperange_cplx((4, nels, 8))
    ad = almmod.AlmData(lmax, alms=alms)
    yield eq_, ad.alms.shape, (4, nels, 8)

def test_init():
    alms = shaperange_cplx((nels,))
    def func():
        ad = almmod.AlmData(lmax=95.0)
    yield assert_raises, TypeError, func
    #Should not be able to set lmax different from alm lmax
    def func():
        ad = almmod.AlmData(lmax=50, alms=alms)
    yield assert_raises, ValueError, func
    def func():
        ad = almmod.AlmData(lmax, alms=4)
    yield assert_raises, TypeError, func
    #Given no alms, should initialize alms of zeros with given lmax
    ad = almmod.AlmData(lmax=lmax)
    yield ok_, np.all(ad.alms == np.zeros(nels))
    #Should not be able to init with a non-complex array
    alms = np.arange(nels)
    def func():
        ad = almmod.AlmData(lmax=lmax, alms=alms)
    yield assert_raises, TypeError, func

def test_assign():
    ad = almmod.AlmData(lmax)
    def func():
        ad.alms = 4
    yield assert_raises, TypeError, func
    #lmax is immutable
    def func():
        ad.lmax = 12
    yield assert_raises, ValueError, func
    #Non-complex
    alms = np.arange(nels)
    def func():
        ad.alms = alms
    yield assert_raises, TypeError, func

def test_shape():
    alms = shaperange_cplx((nels,))
    ad = almmod.AlmData(lmax)
    yield eq_, (nels,), ad.alms.shape
    alms = np.resize(alms, (2, 3, 4, nels))
    ad.alms = alms
    yield eq_, (2, 3, 4, nels), ad.alms.shape
    alms = np.resize(alms, (3, nels, 5))
    ad.alms = alms
    yield eq_, (3, nels, 5), ad.alms.shape

def test_pol():
    #Testing the polarization feature
    def func():
        ad = almmod.AlmData(lmax, pol_axis=0)
    yield assert_raises, ValueError, func
    alms=np.zeros((3, nels), dtype=np.complex)
    def func():
        ad = almmod.AlmData(lmax, alms=alms, pol_axis=1)
    yield assert_raises, ValueError, func
    try:
        ad = almmod.AlmData(lmax, alms=alms, pol_axis=0)
    except:
        raise AssertionError()
    ad = almmod.AlmData(lmax)
    yield eq_, ad.pol_axis, None

def test_appendalms():
    ad = almmod.AlmData(lmax)
    alms = shaperange((1, nels), dtype=np.complex)
    ad.appendalms(alms, along_axis=0)
    combalms = np.append(np.zeros((1, nels)), alms, axis=0)
    yield ok_, np.all(combalms == ad.alms)
    yield eq_, (2, nels), ad.alms.shape
    ad = almmod.AlmData(lmax, alms=alms)
    ad.appendalms(alms, along_axis=0)
    combalms = np.append(alms, alms, axis=0)
    yield ok_, np.all(combalms == ad.alms)
    yield eq_, (2, nels), ad.alms.shape
    #Default axis should be 0:
    ad = almmod.AlmData(lmax, alms=alms)
    ad.appendalms(alms)
    yield ok_, np.all(combalms == ad.alms)
    yield eq_, (2, nels), ad.alms.shape
    alms = shaperange((3, 4, nels, 3, 1), dtype=np.complex)
    ad = almmod.AlmData(lmax, alms=alms)
    def func():
        ad.appendalms(alms, along_axis=2)
    yield assert_raises, ValueError, func
    ad.appendalms(alms, along_axis=4)
    yield eq_, (3, 4, nels, 3, 2), ad.alms.shape

def test_lm2ind():
    lm2inddic = {(6, 4):25, (0, 0):0, (10, 0):55}
    for key, value in lm2inddic.items():
        yield ok_, almmod.lm2ind(key) == value
        yield ok_, almmod.ind2lm(value) == key
    #Test m-major ordering as well:
    lm2inddic = {(3, 2):10, (2, 1):6}
    for key, value in lm2inddic.items():
        yield ok_, almmod.lm2ind(key, lmmax=(4, 4), ordering='m-major') == value
        yield ok_, almmod.ind2lm(value, lmmax=(4, 4), ordering='m-major') == key
    lm2inddic = {(5, 4):23, (0, 0):0, (6, 6):27}
    for key, value in lm2inddic.items():
        yield ok_, almmod.lm2ind(key, lmmax=(6, 6), ordering='m-major') == value
        yield ok_, almmod.ind2lm(value, lmmax=(6, 6), ordering='m-major') == key
    #Test conversion of alms between m-major and l-major ordering
    lmax = 4
    mmax = 4
    nels = lmax * (lmax + 1) // 2 + mmax + 1
    alms = shaperange_cplx((nels,))
    ad = almmod.AlmData(lmax, alms=alms)
    ad.switchordering()
    #l2mdic = {13 : 286, 3593 : 2016, 348 : 1957}
    l2mdic = {6 : 3, 11 : 8}
    for key, value in l2mdic.items():
        yield eq_, int(ad.alms[value].real), key

    lmax = 5
    mmax = 5
    nels = lmax * (lmax + 1) // 2 + mmax + 1
    alms = shaperange_cplx((nels,))
    ad = almmod.AlmData(lmax, alms=alms)
    ad.switchordering()
    l2mdic = {5 : 11, 14 : 18, 20 : 20}
    for key, value in l2mdic.items():
        yield eq_, int(ad.alms[value].real), key


#Testing of cl-class
def test_sanitycl():
    cls = shaperange((lmax + 1,), dtype=float)
    cd = almmod.ClData(lmax)
    yield ok_, cd.lmax == lmax
    cd = almmod.ClData(lmax, cls=cls)
    yield ok_, np.all(cd.cls == cls)
    cls = shaperange_cplx((4, lmax + 1, 8))
    cd = almmod.ClData(lmax, cls=cls)
    yield eq_, cd.cls.shape, (4, lmax + 1, 8)

def test_initcl():
    cls = np.arange(lmax + 1)
    def func():
        cd = almmod.ClData(lmax=95.0)
    yield assert_raises, TypeError, func
    #Should not be able to set lmax different from cl lmax
    def func():
        cd = almmod.ClData(lmax=50, cls=cls)
    yield assert_raises, ValueError, func
    def func():
        cd = almmod.ClData(lmax, cls=4)
    yield assert_raises, TypeError, func
    #Given no cls, should initialize cls of zeros with given lmax
    cd = almmod.ClData(lmax)
    yield ok_, np.all(cd.cls == np.zeros(lmax + 1))
    yield eq_, cd.cls.shape, (lmax + 1,)

def test_assigncl():
    cd = almmod.ClData(lmax)
    def func():
        cd.cls = 4
    yield assert_raises, TypeError, func
    #Lmax is immutable
    def func():
        cd.lmax = 12
    yield assert_raises, ValueError, func

def test_shapecl():
    cls = shaperange((lmax + 1,), dtype=float)
    cd = almmod.ClData(lmax)
    yield eq_, (lmax + 1,), cd.cls.shape
    cls = np.resize(cls, (2, 3, 4, lmax + 1))
    cd.cls = cls
    yield eq_, (2, 3, 4, lmax + 1), cd.cls.shape
    cls = np.resize(cls, (3, lmax + 1, 5))
    cd.cls = cls
    yield eq_, (3, lmax + 1, 5), cd.cls.shape

def test_appendcls():
    cd = almmod.ClData(lmax)
    cls = shaperange((1, lmax + 1))
    cd.appendcls(cls, along_axis=0)
    combcls = np.append(np.zeros((1, lmax + 1)), cls, axis=0)
    yield ok_, np.all(combcls == cd.cls)
    yield eq_, (2, lmax + 1), cd.cls.shape
    cd = almmod.ClData(lmax, cls=cls)
    cd.appendcls(cls, along_axis=0)
    combcls = np.append(cls, cls, axis=0)
    yield ok_, np.all(combcls == cd.cls)
    yield eq_, (2, lmax + 1), cd.cls.shape
    #Default axis should be 0:
    cd = almmod.ClData(lmax, cls=cls)
    cd.appendcls(cls)
    yield ok_, np.all(combcls == cd.cls)
    yield eq_, (2, lmax + 1), cd.cls.shape
    cls = shaperange((3, 4, lmax + 1, 3, 1))
    cd = almmod.ClData(lmax, cls=cls)
    def func():
        cd.appendcls(cls, along_axis=2)
    yield assert_raises, ValueError, func
    cd.appendcls(cls, along_axis=4)
    yield eq_, (3, 4, lmax + 1, 3, 2), cd.cls.shape

def test_speccls():
    alllist = ['TT', 'TE', 'TB', 'EE', 'EB', 'BB']
    telist = ['TT', 'TE', 'EE']
    templist = ['TT']
    #Default: Should assume we only want temperature
    cd = almmod.ClData(lmax)
    yield eq_, cd.nspecs, 1
    yield eq_, cd.spectra, templist
    yield eq_, cd.spec_axis, None
    cd = almmod.ClData(lmax, spectra='all')
    yield eq_, cd.nspecs, 6
    yield eq_, cd.spectra, alllist
    yield eq_, cd.spec_axis, None
    cd = almmod.ClData(lmax, spectra='T-E')
    yield eq_, cd.nspecs, 3
    yield eq_, cd.spectra, telist
    yield eq_, cd.spec_axis, None
    try:
        cd = almmod.ClData(lmax, spectra=['TT', 'BB', 'EB'])
    except:
        raise AssertionError
    #When setting spec_axis, the cl dimension must agree
    cls = shaperange((lmax + 1,))
    cd = almmod.ClData(lmax, spectra='temp')
    def func():
        cd.spec_axis = 0
    yield assert_raises, ValueError, func
    cd = almmod.ClData(lmax, spectra='temp')
    cd.cls = shaperange((1, lmax + 1))
    try:
        cd.spec_axis = 0
    except:
        raise AssertionError
    #Should be possible to change the cls later - i.e. spec_axis should not be
    #imposing anything
    try:
        cd.cls = shaperange((4, 5, lmax + 1)) 
    except:
        raise AssertionError
    def func():
        cd.spec_axis = 1
    yield assert_raises, ValueError, func
    cd.spectra = ['TT', 'TE', 'EE', 'EB', 'BB']
    try:
        cd.spec_axis = 1
    except:
        raise AssertionError

def test_iter():
    alms = shaperange_cplx((3, 4, nels, 3))
    ad = almmod.AlmData(lmax, alms=alms)
    currind = [0, 0, 0]
    indlist = [3, 4, 3]
    for calms in ad:
        yield ok_, np.all(alms[currind[:2] + [Ellipsis,] + currind[2:]] == calms)
        yield ok_, calms.shape == (nels,)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1
    alms = np.arange(nels, dtype=complex)
    md = almmod.AlmData(lmax, alms=alms)
    for calms in md:
        yield ok_, np.all(calms == alms)
    alms = shaperange_cplx((3, nels))
    md = almmod.AlmData(lmax, alms=alms)
    currind = 0
    indlist = 3
    for calms in md:
        yield ok_, np.all(alms[[currind,] + [Ellipsis,]] == calms)
        yield ok_, calms.shape == (nels,)
        currind += 1
    #Keyword pol_iter=True should return (3, nels) or (nels, 3) - array for the
    #iterator
    alms = shaperange_cplx((2, 3, nels))
    ad = almmod.AlmData(lmax, alms=alms, pol_axis=1, pol_iter=True)
    currind = 0
    for calms in ad:
        yield ok_, np.all(alms[[currind,] + [Ellipsis,]] == calms)
        yield ok_, calms.shape == (3, nels)
        currind += 1
    yield eq_, currind, 2
    alms = shaperange_cplx((5, nels, 3))
    ad = almmod.AlmData(lmax, alms=alms, pol_axis=2, pol_iter=True)
    currind = 0
    for calms in ad:
        yield ok_, np.all(alms[[currind,] + [Ellipsis,]] == calms)
        yield ok_, calms.shape == (nels, 3)
        currind += 1
    yield eq_, currind, 5
    alms = shaperange_cplx((5, nels, 6, 3, 3))
    ad = almmod.AlmData(lmax, alms=alms, pol_axis=3, pol_iter=True)
    currind = [0, 0, 0]
    indlist = [5, 6, 3]
    for calms in ad:
        yield ok_, np.all(alms[currind[:1] + [Ellipsis,] + currind[1:2] + 
                        [Ellipsis,] + currind[2:]] == calms)
        yield ok_, calms.shape == (nels, 3)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1
    alms = shaperange_cplx((4, 3, nels, 7, 1))
    ad = almmod.AlmData(lmax, alms=alms, pol_axis=1, pol_iter=True)
    currind = [0, 0, 0]
    indlist = [4, 7, 1]
    for calms in ad:
        yield ok_, np.all(alms[currind[:1] + [Ellipsis,] + currind[1:]] == calms)
        yield ok_, calms.shape == (3, nels)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1


def test_iter_cls():
    cls = shaperange((3, 4, lmax + 1, 3))
    cd = almmod.ClData(lmax, cls=cls)
    currind = [0, 0, 0]
    indlist = [3, 4, 3]
    for ccls in cd:
        yield ok_, np.all(cls[currind[:2] + [Ellipsis,] + currind[2:]] == ccls)
        yield ok_, ccls.shape == (lmax + 1,)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1
    cls = np.arange(lmax+1, dtype=float)
    md = almmod.ClData(lmax, cls=cls)
    for ccls in md:
        yield ok_, np.all(ccls == cls)
    cls = shaperange((3, lmax+1))
    md = almmod.ClData(lmax, cls=cls)
    currind = 0
    indlist = 3
    for ccls in md:
        yield ok_, np.all(cls[[currind,] + [Ellipsis,]] == ccls)
        yield ok_, ccls.shape == (lmax+1,)
        currind += 1
    #Keyword spec_iter=True should return (nspecs, lmax + 1) or 
    #(lmax + 1, nspecs) - array for the iterator
    cls = shaperange((2, 6, lmax + 1))
    cd = almmod.ClData(lmax, cls=cls, spec_axis=1, spec_iter=True, 
                        spectra='all')
    currind = 0
    for ccls in cd:
        yield ok_, np.all(cls[[currind,] + [Ellipsis,]] == ccls)
        yield ok_, ccls.shape == (6, lmax + 1)
        currind += 1
    yield eq_, currind, 2
    cls = shaperange((5, lmax + 1, 3))
    cd = almmod.ClData(lmax, cls=cls, spec_axis=2, spec_iter=True, 
                        spectra='t-e')
    currind = 0
    for ccls in cd:
        yield ok_, np.all(cls[[currind,] + [Ellipsis,]] == ccls)
        yield ok_, ccls.shape == (lmax + 1, 3)
        currind += 1
    yield eq_, currind, 5
    cls = shaperange((5, lmax + 1, 6, 1, 3))
    cd = almmod.ClData(lmax, cls=cls, spec_axis=3, spec_iter=True)
    currind = [0, 0, 0]
    indlist = [5, 6, 3]
    for ccls in cd:
        yield ok_, np.all(cls[currind[:1] + [Ellipsis,] + currind[1:2] + 
                        [Ellipsis,] + currind[2:]] == ccls)
        yield ok_, ccls.shape == (lmax + 1, 1)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1
    cls = shaperange((4, 5, lmax + 1, 7, 1))
    cd = almmod.ClData(lmax, cls=cls, spec_axis=1, spec_iter=True, 
                        spectra = ['TT', 'TE', 'EE', 'BB', 'EB'])
    currind = [0, 0, 0]
    indlist = [4, 7, 1]
    for ccls in cd:
        yield ok_, np.all(cls[currind[:1] + [Ellipsis,] + currind[1:]] == ccls)
        yield ok_, ccls.shape == (5, lmax + 1)
        trace_ind = 2
        while indlist[trace_ind] == currind[trace_ind] + 1 and trace_ind != 0:
            currind[trace_ind] = 0
            trace_ind -= 1
        currind[trace_ind] += 1

def test_operators():
    ad = almmod.AlmData(lmax=lmax, alms=alms)
    ad2 = almmod.AlmData(lmax=lmax, alms=alms)
    yield ok_, np.all(ad.alms + ad2.alms == (ad + ad2).alms)
    yield ok_, np.all(ad.alms + ad.alms == (ad + ad).alms)
    yield ok_, np.all(ad.alms * ad2.alms == (ad * ad2).alms)
    yield ok_, np.all(ad.alms - ad2.alms == (ad - ad2).alms)
    ad.alms = ad.alms + 1
    ad2.alms = ad.alms + 1
    yield ok_, np.all(ad.alms / ad2.alms == (ad / ad2).alms)
    yield eq_, ad.alms[67], ad[67]
    ad[87] = 245.23
    yield eq_, ad[87], 245.23
    yield eq_, ad.shape, ad.alms.shape
