from __future__ import division
import utils
import fileutils
import numpy as np
from nose.tools import ok_, eq_, assert_raises
import sys

def test_alm2map():
    try:
        ad = fileutils.read_file('eirikalms_l95.fits')
        md = utils.alm2map(ad, nside=32)
    except:
        raise AssertionError()
