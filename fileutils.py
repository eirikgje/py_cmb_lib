from __future__ import division
import numpy as np
import pyfits
import mapmod

def read_fits_map(fname):
    #This will expand as need arises, for now, pretty ad-hoc
    hdulist = pyfits.open(fname)
    #TODO: Check whether saved as image or binary table
    data = hdulist[1].data
    cols = hdulist[1].columns
    hdr = hdulist[1].header
    #subd = np.array(len(cols.names))
    nside = hdr['nside']
    npix = 12*nside*nside
    map = np.zeros((subd, npix))
    for i in range(subd):
        map[i] = data.field(i).flatten()
    #md.map = map
    md = mapmod.MapData(nside, map=map, ordering=hdr['ordering'], 
    hdulist.close()
    return md

def write_fits_map(fname, md, bintab=True):
    #So far will only write information that is actually needed for this
    #library. Could cause compatibility problems later.
    #if bintab:
