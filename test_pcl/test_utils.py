from __future__ import division
import utils
import fileutils
import numpy as np
from nose.tools import ok_, eq_, assert_raises
import sys

#def test_a2m2a():
#    try:
#        ad = fileutils.read_file('eirikalms_l95.fits')
#        md = utils.alm2map(ad, nside=32)
#    except:
#        raise AssertionError()
#
#    try:
#        md = fileutils.read_file('eirikmap_n32_new.fits')
#        weights = fileutils.read_file('weight_ring_n00032.fits')
#        ad = utils.map2alm(md, lmax = 95, mmax = 95, weights)
#    except:
#        raise AssertionError()
