from __future__ import division
import beammod
import numpy as np
from nose.tools import ok_, eq_, assert_raises

def shaperange(shape, dtype=float):
    m = 1
    for s in shape:
        m = m * s
    srange = np.arange(m, dtype=dtype)
    srange = srange.reshape(shape)
    return srange.copy()

def test_gaussbeam():
    beam = beammod.gaussbeam(10.0, 95, ndim=1)
    yield eq_, ['TT'], beam.spectra
    beam = beammod.gaussbeam(10.0, 95, ndim=4)
    yield eq_, ['TT', 'EE', 'BB', 'TE'], beam.spectra

lmax = 90
nels = lmax + 1
beam = np.arange(nels)
def test_beamdata_sanity():
    beam = np.arange(nels)
    bd = beammod.BeamData(lmax)
    yield eq_,  bd.lmax, lmax
    yield eq_, bd.beam.shape, (nels,)
    bd = beammod.BeamData(lmax, beam=beam)
    yield eq_,  bd.lmax, lmax
    yield eq_, bd.beam.shape, (nels,)
    yield ok_, np.all(bd.beam == beam)
    def func():
        bd = beammod.BeamData(lmax, beam=beam, beam_axis=1)
    yield assert_raises, IndexError, func
    beam = shaperange((nels, 3))
    bd = beammod.BeamData(lmax, beam=beam)
    yield eq_, bd.lmax, lmax
    yield eq_, bd.beam.shape, beam.shape
    yield ok_, np.all(bd.beam == beam)
    yield eq_, bd.beam_axis, 0
