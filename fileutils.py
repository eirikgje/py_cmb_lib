from __future__ import division
import numpy as np
import pyfits
#import mapmod

def read_fits_map(fname, map, nside, ordering):
    #This will expand as need arises, for now, pretty ad-hoc
    hdulist = pyfits.open(fname)
    data = hdulist[1].data
    cols = hdulist[1].columns
    subd = len(cols.names)
    map.subdivide(subd)
    for i in range(subd):
        map = d.field(i).flatten()

    
