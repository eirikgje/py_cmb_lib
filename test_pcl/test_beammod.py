from __future__ import division
import beammod
import numpy as np
from nose.tools import ok_, eq_, assert_raises

def test_gaussbeam():
    beam = beammod.gaussbeam(10.0, 95, ndim=1)
    yield eq_, ['TT'], beam.spectra
    beam = beammod.gaussbeam(10.0, 95, ndim=4)
    yield eq_, ['TT', 'EE', 'BB', 'TE'], beam.spectra
