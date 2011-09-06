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
                    objdata.pol_axis = 0
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
                    pol_axis = 0
                else:
                    raise NotImplementedError()
                    shape = (lmax * (lmax + 1) // 2 + lmax + 1,)
                    pol_axis = None
                objdata = almmod.AlmData(lmax, alms = np.zeros(shape, 
                                         dtype=np.complex), pol_axis=pol_axis)
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
            elif exthdr['EXTNAME'] == 'PIXEL WINDOW':
                return 'cls'
        if 'MAX-LPOL' in exthdr:
            if 'MAX-MPOL' in exthdr:
                return 'alms'
            else:
                return 'cls'
        if 'NSIDE' in exthdr:
            if 'MAX-MPOL' in exthdr:
                return 'alms'
            elif 'MAX-LPOL' in exthdr:
                return 'cls'
            else:
                return 'map'
    else:
        raise NotImplementedError()
    raise TypeError("Cannot determine fits data type")

def write_file(fname, data, bintab=True):
    if fname.endswith('.fits'):
        if isinstance(data, mapmod.MapData):
            if bintab:
                if data.nside <= 8:
                    raise NotImplementedError("""Nside must be at least 16 at 
                                                this point""")
                npix = 12*data.nside**2
                if npix % 1024 != 0:
                    raise NotImplementedError("npix must be a multiple of 1024")
                numrows = npix // 1024
                if data.pol_axis is not None:
                    pol = True
                    if len(data.map.shape) != 2:
                        raise NotImplementedError()
                    cols = []
                    for i in range(3):
                        if i == 0:
                            name = 'TEMPERATURE'
                        elif i == 1:
                            name = 'Q-POLARISATION'
                        elif i == 2:
                            name = 'U-POLARISATION'
                        cols.append(pyfits.Column(name=name, format='1024E', 
                                            array = data.map[(Ellipsis,) * 
                                            data.pol_axis + (i,) + (Ellipsis,) *
                                            (data.map.ndim - 1 - 
                                            data.pol_axis)].reshape(numrows, 
                                                                   1024)))
                else:
                    pol = False
                    if (len(data.map.shape)) != 1:
                        raise NotImplementedError()
                    cols = [pyfits.Column(name='TEMPERATURE', format='1024E',
                                        array=data.map.reshape(numrows, 1024))]
                cols = pyfits.ColDefs(cols)
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
            else:
                raise NotImplementedError()
        elif isinstance(data, almmod.ClData):
            if bintab:
                raise NotImplementedError()
            else:
                if (data.spec_axis is None or (data.cls.shape[data.spec_axis] 
                    == 1 and data.spectra == ['TT'])):
                    pol = False
                else:
                    pol = True
                if data.cls.ndim == 1 or (data.cls.ndim == 2 
                        and data.spec_axis is not None):
                    cols = []
                    for i in range(len(data.spectra)):
                        if data.spectra[i] == 'TT':
                            name = 'TEMPERATURE'
                        elif data.spectra[i] == 'TE':
                            name = 'G-T'
                        elif data.spectra[i] == 'EE':
                            name = 'GRADIENT'
                        elif data.spectra[i] == 'BB':
                            name = 'CURL'
                        elif data.spectra[i] == 'EB':
                            raise NotImplementedError()
                        elif data.spectra[i] == 'TB':
                            raise NotImplementedError()
                        if data.spec_axis is None:
                            array = data.cls
                        else:
                            array = data.cls[(Ellipsis,) * data.spec_axis + (i,)
                                    + (Ellipsis,) * (data.cls.ndim - 1 - 
                                        data.spec_axis)]
                        cols.append(pyfits.Column(name=name, format = 'E24.15', 
                                        array = array))
                    cols = pyfits.ColDefs(cols, tbtype="TableHDU")
                    prihdu = pyfits.PrimaryHDU()
                    tbhdu = pyfits.new_table(cols, tbtype='TableHDU')
                    thdr = tbhdu.header
                    thdr.update('EXTNAME', 'POWER SPECTRUM', 
                                'Power spectrum : C(l)')
                    thdr.update('POLAR', pol,
                                'Polarisation included (True/False)')
                    thdr.update('MAX-LPOL', data.lmax, 'Maximum L multipole')
                    thdulist = pyfits.HDUList([prihdu, tbhdu])
                else:
                    raise NotImplementedError()
        else:
            raise NotImplementedError()
        thdulist.writeto(fname)
    else:
        raise NotImplementedError()
