from __future__ import division
import numpy as np
import pyfits
import mapmod
import almmod

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
    md = mapmod.MapData(nside, ordering=hdr['ordering'], pol=hdr['polar'])
    for i in range(md.map.shape[1]):
        md.map[0, i] = data.field(i).flatten()
    hdulist.close()
    return md

def write_fits_map(fname, md, bintab=True):
    #So far will only write information that is actually needed for this
    #library. Could cause compatibility problems later.
    #So far only writes a single map with or without polarisation.
    if bintab:
        if md.nside <= 8:
            raise NotImplementedError("Nside must be at least 16 at this point")
        npix = 12*md.nside**2
        if npix % 1024 != 0:
            raise NotImplementedError("npix must be a multiple of 1024")
        numrows = npix // 1024
        if md.pol:
            if len(md.map.shape) != 3:
                raise NotImplementedError()
            col1 = pyfits.Column(name='TEMPERATURE', format='1024E', 
                    array = md.map[0, 0].reshape(numrows, 1024))
            col2 = pyfits.Column(name='Q-POLARISATION', format='1024E', 
                    array = md.map[0, 1].reshape(numrows, 1024))
            col3 = pyfits.Column(name='U-POLARISATION', format='1024E', 
                    array = md.map[0, 2].reshape(numrows, 1024))
            cols = pyfits.ColDefs([col1, col2, col3])
        elif not md.pol:
            if (len(md.map.shape)) != 2:
                raise NotImplementedError()
            col1 = pyfits.Column(name='TEMPERATURE', format='1024E',
                                 array=md.map[0].reshape(numrows, 1024))
            cols = pyfits.ColDefs([col1])
        prihdu = pyfits.PrimaryHDU()
        tbhdu = pyfits.new_table(cols)
        thdr = tbhdu.header
        thdr.update('NSIDE', md.nside, 'Resolution parameter for HEALPIX')
        thdr.update('ORDERING', md.ordering.upper(), 
                    'Pixel ordering scheme, either RING or NESTED')
        thdr.update('POLAR', md.pol, 'Polarisation included (True/False)')
        thdulist = pyfits.HDUList([prihdu, tbhdu])
        thdulist.writeto(fname)
    else:
        raise NotImplementedError()

def read_fits_alms(fname):
    hdulist = pyfits.open(fname)
    data = hdulist[1].data
    header = hdulist[1].header
    pol = header['polar']
    lmax = header['max-lpol']
    mmax = header['max-mpol']
    ad = almmod.AlmData(lmax, pol=pol)
    if pol:
        for i in range(1, 4):
            real = hdulist[i].data.field('REAL')
            imag = hdulist[i].data.field('IMAG')
            ad.alms[0, i-1] = real + 1j * imag
    hdulist.close()
    return ad
