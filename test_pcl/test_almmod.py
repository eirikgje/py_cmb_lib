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
        out[i] = np.complex(2*i, 2*i+1)
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
    def func():
        a = almmod.ind2lm(3.0)
    yield assert_raises, TypeError, func
    def func():
        a = almmod.lm2ind(3)
    yield assert_raises, TypeError, func
    def func():
        a = almmod.lm2ind((3.0, 4))
    yield assert_raises, TypeError, func
    def func():
        a = almmod.ind2lm((4, 3))
    yield assert_raises, TypeError, func

#Testing of cl-class
def test_sanitycl():
    cls = shaperange((lmax + 1,), dtype=float)
    cd = almmod.ClData(lmax)
    yield ok_, cd.lmax == lmax
    cd = almmod.ClData(lmax, cls=cls)
    yield ok_, np.all(cd.cls == cls)
    cls = shaperange_cplx((4, lmax + 1, 8))
    ad = almmod.ClData(lmax, cls=cls)
    yield eq_, ad.cls.shape, (4, lmax + 1, 8)

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
