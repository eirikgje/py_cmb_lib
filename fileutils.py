from __future__ import division
import numpy as np
import pyfits
#import mapmod

def read_fits_map(fname, md):
    #This will expand as need arises, for now, pretty ad-hoc
    hdulist = pyfits.open(fname)
    data = hdulist[1].data
    cols = hdulist[1].columns
    hdr = hdulist[1].header
    subd = np.array(len(cols.names))
    nside = hdr['nside']
    if md.nside is None:
        md.nside = nside
    else:
        if md.nside != nside:
            raise ValueError("""MapData nside is not the same as file nside""")
    if md.subd is None:
        md.subdivide(subd)
    else:
        if not md.subd == subd:
            raise ValueError("""MapData object is already subdivided, and it
                                has the wrong subdivision""")
    if md.ordering is None:
        md.ordering = hdr['ordering']
    else:
        if md.ordering != hdr['ordering'].lower:
            md.switchordering()
    npix = 12*nside*nside
    map = np.zeros((subd, npix))
    for i in range(subd):
        map[i] = data.field(i).flatten()
    md.map = map
    hdulist.close()
    return md

def write_fits_map(fname, md):
    #So far will only write information that is actually needed for this
    #library. Could cause compatibility problems later.
