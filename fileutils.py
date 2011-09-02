from __future__ import division
import numpy as np
import pyfits
import mapmod
import almmod

def read_file(fname, type=None):
    if fname.endswith('.fits'):
        #This will expand as need arises, for now, pretty ad-hoc
        hdulist = pyfits.open(fname)
        if type is None:
            type = determine_type_fits(hdulist)
        if type.lower() == 'map':
            if len(hdulist) == 2:
                data = hdulist[1].data
                cols = hdulist[1].columns
                hdr = hdulist[1].header
                nside = hdr['nside']
                npix = 12*nside*nside
                if hdr['polar']:
                    shape = (3, npix)
                else:
                    shape = (npix,)
                objdata = mapmod.MapData(nside, ordering=hdr['ordering'], 
                                        map=np.zeros(shape))
                if hdr['polar']:
                    for i in range(3):
                        objdata.map[i] = data.field(i).flatten()
                    objdata.polaxis = 0
                else:
                    objdata.map = data.field(i).flatten()
            else:
                raise NotImplementedError()
        elif type.lower() == 'alms':
            if len(hdulist) == 4:
                data = hdulist[1].data
                hdr = hdulist[1].header
                pol = hdr['polar']
                lmax = hdr['max-lpol']
                mmax = hdr['max-mpol']
                if hdr['polar']:
                    shape = (3, lmax * (lmax + 1) // 2 + lmax + 1)
                else:
                    raise NotImplementedError()
                    shape = (lmax * (lmax + 1) // 2 + lmax + 1,)
                objdata = almmod.AlmData(lmax, alms = np.zeros(shape, 
                                         dtype=np.complex))
                if hdr['polar']:
                    for i in range(1, 4):
                        real = hdulist[i].data.field('REAL')
                        imag = hdulist[i].data.field('IMAG')
                        objdata.alms[i-1] = real + 1j * imag
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        hdulist.close()
    else:
        raise NotImplementedError()
    return objdata

def determine_type_fits(hdulist):
    if hdulist[0].data is None:
        #Assume extension
        exthdr = hdulist[1].header
        if 'EXTNAME' in exthdr:
            if exthdr['EXTNAME'] == 'FULL SKY MAP':
                return 'map'
            elif exthdr['EXTNAME'] == 'HEALPIX MAP':
                return 'map'
            elif exthdr['EXTNAME'] == 'POWER SPECTRUM':
                return 'cls'
            elif exthdr['EXTNAME'].startswith('ANALYSED A_LMS'):
                return 'alms'
        if 'NSIDE' in exthdr:
            if 'MAX-MPOL' in exthdr:
                return 'alms'
            else:
                return 'map'
        if 'MAX-LPOL' in exthdr:
            if 'MAX-MPOL' in exthdr:
                return 'alms'
            else:
                return 'cls'
    else:
        raise NotImplementedError()
    raise TypeError("Cannot determine fits data type")

def write_file(fname, data, bintab=True):
    if fname.endswith('.fits'):
        #Write fits file
        if isinstance(data, mapmod.MapData):
            #Write map
            if bintab:
                if data.nside <= 8:
                    raise NotImplementedError("""Nside must be at least 16 at 
                                                this point""")
                npix = 12*data.nside**2
                if npix % 1024 != 0:
                    raise NotImplementedError("npix must be a multiple of 1024")
                numrows = npix // 1024
                if data.polaxis is not None:
                    pol = True
                    if len(data.map.shape) != 2:
                        raise NotImplementedError()
                    col1 = pyfits.Column(name='TEMPERATURE', format='1024E', 
                            array = data.map[(Ellipsis,) * data.polaxis + (0,) 
                                             + (Ellipsis,) * (data.map.ndim - 1
                                             - data.polaxis)].reshape(numrows, 
                                                                      1024))
                    col2 = pyfits.Column(name='Q-POLARISATION', format='1024E', 
                            array = data.map[(Ellipsis,) * data.polaxis + (1,) 
                                             + (Ellipsis,) * (data.map.ndim - 1
                                             - data.polaxis)].reshape(numrows, 
                                                                      1024))
                    col3 = pyfits.Column(name='U-POLARISATION', format='1024E', 
                            array = data.map[(Ellipsis,) * data.polaxis + (2,) 
                                             + (Ellipsis,) * (data.map.ndim - 1
                                             - data.polaxis)].reshape(numrows, 
                                                                      1024))
                    cols = pyfits.ColDefs([col1, col2, col3])
                else:
                    pol = False
                    if (len(data.map.shape)) != 1:
                        raise NotImplementedError()
                    col1 = pyfits.Column(name='TEMPERATURE', format='1024E',
                                        array=data.map.reshape(numrows, 1024))
                    cols = pyfits.ColDefs([col1])
                prihdu = pyfits.PrimaryHDU()
                tbhdu = pyfits.new_table(cols)
                thdr = tbhdu.header
                thdr.update('NSIDE', data.nside, 
                            'Resolution parameter for HEALPIX')
                thdr.update('ORDERING', data.ordering.upper(), 
                            'Pixel ordering scheme, either RING or NESTED')
                thdr.update('POLAR', pol, 'Polarisation included (True/False)')
                thdr.update('EXTNAME', 'HEALPIX MAP', 'HEALPix map')
                thdulist = pyfits.HDUList([prihdu, tbhdu])
                thdulist.writeto(fname)
            else:
                raise NotImplementedError()
        elif isinstance(data, mapmod.ClData):
            #Write Cl file
            pass
    else:
        raise NotImplementedError()
