from __future__ import division
import almmod
import numpy as np
from nose.tools import ok_, eq_, assert_raises
import sys

#nside = 32
lmax = 95
numind = lmax * (lmax + 1) / 2 + lmax + 1

#npix = 12 * nside ** 2
def shaperange(shape):
    m = 1
    for s in shape:
        m = m * s
    srange = np.arange(m, dtype='float32')
    srange.reshape(shape)
    return srange

#def test_map2alm():

def test_sanity():
    ad = almmod.AlmData(lmax)
    yield ok_, ad.lmax == lmax
    alms = np.arange(numind)
    ad = almmod.AlmData(lmax, alms=alms)
    yield ok_, np.all(ad.alms == alms)
    ad = almmod.AlmData(lmax, lsubd=(3, 2, 1))
    yield ok_, np.all((3, 2, 1) == ad.subd)

