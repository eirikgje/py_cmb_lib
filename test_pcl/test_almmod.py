from __future__ import division
import almmod
import numpy as np
from nose.tools import ok_, eq_, assert_raises
import sys

lmax = 95
nels = lmax * (lmax + 1) / 2 + lmax + 1
alms = np.arange(2*nels)
alms = alms.reshape((nels, 2))

def shaperange(shape):
    m = 1
    for s in shape:
        m = m * s
    srange = np.arange(m, dtype='float32')
    srange = srange.reshape(shape)
    return srange.copy()

#def test_alms():

def test_sanity():
    ad = almmod.AlmData(lmax)
    yield ok_, ad.lmax == lmax
    ad = almmod.AlmData(lmax, alms=alms)
    yield ok_, np.all(ad.alms == alms)
    ad = almmod.AlmData(lmax, lsubd=(3, 2, 1))
    yield ok_, np.all((3, 2, 1) == ad.subd)

def test_init():
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
    yield ok_, np.all(ad.alms == np.zeros((1, nels, 2)))
    ad = almmod.AlmData(lmax, lsubd=(3, 4))
    yield eq_, (3, 4, 1, nels, 2), ad.alms.shape
    ad = almmod.AlmData(lmax, rsubd=(2, 5))
    yield eq_, (1, 2, 5, nels, 2), ad.alms.shape
    ad = almmod.AlmData(lmax, lsubd=6, rsubd=(2, 5))
    yield eq_, (6, 1, 2, 5, nels, 2), ad.alms.shape
    ad = almmod.AlmData(lmax, lsubd=(6, 3), rsubd=2)
    yield eq_, (6, 3, 1, 2, nels, 2), ad.alms.shape

def test_assign():
    ad = almmod.AlmData(lmax)
    def func():
        ad.alms = 4
    yield assert_raises, TypeError, func
    def func():
        ad.lmax = 12.0
    yield assert_raises, TypeError, func
    #After subdividing, should be possible to assign alms of the given size
    ad = almmod.AlmData(lmax)
    ad.subdivide(3)
    alms = np.zeros((3, nels, 2))
    def func():
        ad.alms = alms

    ad = almmod.AlmData(lmax)
    ad.subdivide((3, 4), left_of_dyn_d=False)
    alms = np.zeros((3, 4, nels, 2))
    try:
        ad.alms = alms
    except:
        raise AssertionError()
    alms = np.zeros((1, 3, 4, nels, 2))
    try:
        ad.alms = alms
    except:
        raise AssertionError()

def test_shape():
    ad = almmod.AlmData(lmax)
    yield eq_, (1, nels, 2), ad.alms.shape
    ad.subdivide(5)
    yield eq_, (5, 1, nels, 2), ad.alms.shape
    alms = shaperange((nels, 2))
    def func():
        ad.alms = alms
    yield assert_raises, ValueError, func
    alms.resize((2, 3, 4, nels, 2))
    def func():
        ad.alms = alms
    yield assert_raises, ValueError, func
    ad = almmod.AlmData(lmax, lsubd=(3, 2))
    yield eq_, (3, 2, 1, nels, 2), ad.alms.shape
    #Should be possible to choose whether the subdivision should be to the
    #left or right of the dynamical dimension
    ad = almmod.AlmData(lmax)
    ad.subdivide(3, left_of_dyn_d=False)
    yield eq_, (1, 3, nels, 2), ad.alms.shape
    ad = almmod.AlmData(lmax)
    ad.subdivide((3, 5), left_of_dyn_d=False)
    yield eq_, (1, 3, 5, nels, 2), ad.alms.shape
    ad = almmod.AlmData(lmax)
    ad.subdivide((3, 5))
    yield eq_, (3, 5, 1, nels, 2), ad.alms.shape
    ad = almmod.AlmData(lmax)
    ad.subdivide((3, 5), left_of_dyn_d=False)
    ad.subdivide(4)
    yield eq_, (4, 1, 3, 5, nels, 2), ad.alms.shape

def test_pol():
    #Testing the polarization feature
    ad = almmod.AlmData(lmax, pol=True)
    yield eq_, (1, 3, nels, 2), ad.alms.shape
    alms = np.reshape(np.arange(3*nels*2), (1, 3, nels, 2))
    ad.alms = alms
    yield ok_, np.all(alms == ad.alms)
    ad = almmod.AlmData(lmax)
    ad.pol = True
    yield eq_, (1, 3, nels, 2), ad.alms.shape
    #'pol' keyword is only supposed to be a 'compatibility flag' - i.e., if 
    #the AlmData object already is compatible, then do nothing
    #alms = np.reshape(np.arange(3*nels), (1, 3, nels))
    alms = shaperange((1, 3, nels, 2))
    ad = almmod.AlmData(lmax, rsubd=3, alms=alms)
    alms = ad.alms
    ad.pol = True
    yield ok_, np.all(alms == ad.alms)
    #Other way around:
    ad = almmod.AlmData(lmax, pol=True)
    alms = ad.alms
    ad.pol = False
    yield ok_, np.all(alms == ad.alms)

def test_appendalms():
    ad = almmod.AlmData(lmax)
    alms = shaperange((nels, 2))
    ad.appendalms(alms)
    combalms = np.append(np.zeros((nels, 2)), alms)
    combalms = np.reshape(combalms, (2, nels, 2))
    yield ok_, np.all(combalms == ad.alms)
    yield eq_, (2, nels, 2), ad.alms.shape
    ad = almmod.AlmData(lmax)
    alms = shaperange((1, nels, 2))
    ad.appendalms(alms)
    combalms = np.append(np.zeros((nels, 2)), alms)
    combalms = np.reshape(combalms, (2, nels, 2))
    yield ok_, np.all(combalms == ad.alms)
    yield eq_, (2, nels, 2), ad.alms.shape
    ad = almmod.AlmData(lmax, lsubd=(3, 2))
    alms = shaperange((3, 2, nels, 2))
    ad.appendalms(alms)
    alms = alms.reshape((3, 2, 1, nels, 2))
    combalms = np.append(np.zeros((3, 2, 1, nels, 2)), alms, axis=2)
    yield ok_, np.all(combalms == ad.alms)
    yield eq_, (3, 2, 2, nels, 2), ad.alms.shape
    ad = almmod.AlmData(lmax, rsubd=8)
    alms = shaperange((8, nels, 2))
    ad.appendalms(alms)
    alms = alms.reshape((1, 8, nels, 2))
    combalms = np.append(np.zeros((1, 8, nels, 2)), alms, axis=0)
    yield ok_, np.all(combalms == ad.alms)
    yield eq_, (2, 8, nels, 2), ad.alms.shape
    ad = almmod.AlmData(lmax, lsubd=3, rsubd=(8, 10))
    alms = shaperange((3, 8, 10, nels, 2))
    ad.appendalms(alms)
    alms = alms.reshape((3, 1, 8, 10, nels, 2))
    ad.appendalms(alms)
    combalms = np.append(np.zeros((3, 1, 8, 10, nels, 2)), alms, axis=1)
    combalms = np.append(combalms, alms, axis=1)
    yield ok_, np.all(combalms == ad.alms)
    yield eq_, (3, 3, 8, 10, nels, 2), ad.alms.shape
    #Should be possible to append a AlmData object as well
    ad = almmod.AlmData(lmax)
    ad2 = almmod.AlmData(lmax)
    ad.appendalms(ad2)
    yield eq_, (2, nels, 2), ad.alms.shape
    ad2 = almmod.AlmData(lmax*2)
    def func():
        ad.appendalms(ad2)
    yield assert_raises, ValueError, func
