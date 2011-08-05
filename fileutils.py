from __future__ import division
import numpy as np
import pyfits

def read_fits_map(fname, map, nside, ordering):
    hdulist = pyfits.open(fname)
