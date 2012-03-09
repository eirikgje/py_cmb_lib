from __future__ import division
import numpy as np
import pyfits
import mapmod
import almmod
import beammod

_HPIX_BAD_DATA = -1.6375e30
_HPIX_MODIFY_VAL = -4e29

def read_file(fname, type=None):
    if fname.endswith('.fits'):
        #This will expand as need arises, for now, pretty ad-hoc
        hdulist = pyfits.open(fname)
        if type is None:
            type = determine_type_fits(hdulist, fname)
        if type.lower() == 'map':
            if len(hdulist) == 2:
                data = hdulist[1].data
                cols = hdulist[1].columns
                hdr = hdulist[1].header
                nside = hdr['nside']
                npix = 12*nside*nside
                if 'polar' in hdr:
                    if hdr['polar']:
                        shape = (3, npix)
                    else:
                        if hdr['naxis2'] != 1:
                            if hdr['tform1'] == '1024E':
                                shape = (npix,)
                            else:
                                shape = (hdr['naxis2'], npix)
                        else:
                            shape = (npix,)
                else:
                    shape = (hdr['TFIELDS'], npix)
                    
                objdata = mapmod.MapData(nside, ordering=hdr['ordering'], 
                                        map=np.zeros(shape))
                if 'polar' in hdr:
                    if hdr['polar']:
                        for i in range(3):
                            objdata.map[i] = data.field(i).flatten()
                        objdata.pol_axis = 0
                    else:
                        if hdr['naxis2'] != 1:
                            if cols[0].unit in ('K_CMB', 'K'):
                                fac = 1e6
                            elif cols[0].unit in ('(K_CMB)^2',):
                                fac = 1e12
                            elif cols[0].unit in ('N_hit', 'unknown', 'counts'):
                                fac = 1
                            elif cols[0].unit == 'muK':
                                fac = 1
                            elif cols[0].unit == 'mK':
                                fac = 1e3
                            elif cols[0].unit is None:
                                fac = 1
                            else:
                                print cols[0].unit
                                raise ValueError("Unknown unit")
                            if hdr['tform1'] == '1024E':
                                objdata.map[:] = data.field(0).flatten().astype(np.float64) * fac
                            else:
                                for i in range(hdr['naxis2']):
                                    print objdata.map[i].shape
                                    print data.field(0)[i].shape
                                    objdata.map[i] = data.field(0)[i].astype(np.float64)*fac
                        else:
                            for i in range(hdr['TFIELDS']):
                            #Saves the data in muK as default
                                if cols[i].unit in ('K_CMB', 'K'):
                                    fac = 1e6
                                elif cols[i].unit in ('(K_CMB)^2',):
                                    fac = 1e12
                                elif cols[i].unit in ('N_hit', 'unknown', 'counts'):
                                    fac = 1
                                elif cols[i].unit == 'muK':
                                    fac = 1
                                elif cols[0].unit == 'mK':
                                    fac = 1e3
                                elif cols[i].unit is None:
                                    fac = 1
                                else:
                                    print cols[i].unit
                                    raise ValueError("Unknown unit")
                                objdata.map[i] = (data.field(i).flatten()).astype(np.float64)*fac
                                objdata.map[i, data.field(i).flatten() == 
                                            float(hdr['bad_data'])] = np.nan
                #                        objdata.map[i, objdata.map[i] > 
#                                    float(hdr['bad_data'])] = \
#                                    objdata.map[i, objdata.map[i] > 
#                                    float(hdr['bad_data'])] * fac

                else:
                    for i in range(hdr['TFIELDS']):
                        #Saves the data in muK as default
                        if cols[i].unit in ('K_CMB', 'K'):
                            fac = 1e6
                        elif cols[i].unit in ('(K_CMB)^2',):
                            fac = 1e12
                        elif cols[i].unit in ('N_hit', 'unknown', 'counts'):
                            fac = 1
                        elif cols[i].unit == 'muK':
                            fac = 1
                        elif cols[i].unit == 'mK':
                            fac = 1e3
                        elif cols[i].unit is None:
                            fac = 1
                        else:
                            raise ValueError("Unknown unit")
                        objdata.map[i] = (data.field(i).flatten()).astype(np.float64)*fac
#                        objdata.map[i, data.field(i).flatten() == 
#                                    float(hdr['bad_data'])] = \
#                                    float(hdr['bad_data'])
                        objdata.map[i, data.field(i).flatten() <
                                    _HPIX_BAD_DATA - _HPIX_MODIFY_VAL] = np.nan
#                        objdata.map[i, objdata.map[i] > 
#                                    float(hdr['bad_data'])] = \
#                                    objdata.map[i, objdata.map[i] > 
#                                    float(hdr['bad_data'])] * fac
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
        elif type.lower() == 'weight_ring':
            if len(hdulist) == 2:
                data = hdulist[1].data
                hdr = hdulist[1].header
                objdata = np.zeros((3, hdr['NAXIS2']))
                for i in range(3):
                    objdata[i] = data.field(i)
                objdata = objdata + 1
            else:
                raise NotImplementedError()
        elif type.lower() == 'beam':
            if len(hdulist) == 2:
                data = hdulist[1].data
                hdr = hdulist[1].header
                if hdr['polar']:
                    shape = (3, hdr['NAXIS2'])
                else:
                    shape = (1, hdr['NAXIS2'])
                if hdr['polar']:
                    objdata = beammod.BeamData(lmax=hdr['NAXIS2'] - 1,
                                            beam=np.zeros(shape),
                                            pol_axis=0)
                else:
                    objdata = beammod.BeamData(lmax=hdr['NAXIS2'] - 1,
                                            beam=np.zeros(shape))
                for i in range(shape[0]):
                    objdata.beam[i, :] = data.field(i)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        hdulist.close()
    else:
        raise NotImplementedError()
    return objdata

def determine_type_fits(hdulist, fname):
    if fname.startswith('weight_ring_n'):
        return 'weight_ring'
    if fname.startswith('pixel_window_n'):
        return 'cls'

    if len(hdulist) >= 2:
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
                return 'beam'
            elif exthdr['EXTNAME'] == 'SIMULATED MAP':
                return 'map'
            elif exthdr['EXTNAME'] == 'BEAM':
                return 'beam'
        if 'CREATOR' in exthdr:
            if exthdr['CREATOR'] == 'QUAD_RING':
                return 'weight_ring'
        if 'WEIGHTS' in exthdr['TTYPE1']:
            return 'weight_ring'


        #If things progress down here, it gets a little dirty
        if 'MAX-LPOL' in exthdr:
            if 'MAX-MPOL' in exthdr:
                return 'alms'
            else:
                return 'cls'
        if 'NSIDE' in exthdr:
            if 'MAX-MPOL' in exthdr:
                return 'alms'
            elif 'MAX-LPOL' in exthdr:
                if exthdr['NAXIS2'] == 2*exthdr['NSIDE']:
                    return 'weight_ring'
                else:
                    return 'cls'
            else:
                return 'map'
    elif 'beam' in fname.lower():
        return 'beam'
    else:
        raise NotImplementedError()
    raise TypeError("Cannot determine fits data type")

def write_file(fname, data, bintab=True, table_header=None, divide_data=False, 
                names=None, bad_data=None, sig=None):
    if fname.endswith('.fits'):
        if isinstance(data, mapmod.MapData):
            if bintab:
                if divide_data:
                    #This should preferrably not be used, it is obsolete. 
                    #It divides the map into chunks of 1024 pixels and makes a 
                    #new record for each chunk, but also assumes a specific 
                    #shape for the input map
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
                                        data.pol_axis)].reshape(numrows, 1024)))
                    else:
                        pol = False
                        if (len(data.map.shape)) != 1:
                            raise NotImplementedError()
                        cols = [pyfits.Column(name='TEMPERATURE', 
                                              format='1024E', 
                                              array=data.map.reshape(numrows, 
                                              1024))]
                    cols = pyfits.ColDefs(cols)
                    prihdu = pyfits.PrimaryHDU()
                    tbhdu = pyfits.new_table(cols)
                    thdr = tbhdu.header
                    thdr.update('NSIDE', data.nside, 
                                'Resolution parameter for HEALPIX')
                    thdr.update('ORDERING', data.ordering.upper(), 
                                'Pixel ordering scheme, either RING or NESTED')
                    thdr.update('POLAR', pol, 
                                'Polarisation included (True/False)')
                    thdr.update('EXTNAME', 'HEALPIX MAP', 'HEALPix map')
                    thdulist = pyfits.HDUList([prihdu, tbhdu])
                else:
                    if len(data.map.shape) == 1:
                        nmaps = 1
                    else:
                        nmaps = sum(data.map.shape[:data.pix_axis]) + \
                                sum(data.map.shape[data.pix_axis + 1:])
                    if names is None:
                        names = ["MAP%1d" % i for i in range(nmaps)]
                    else:
                        if len(names) != nmaps:
                            raise ValueError("Length of name array must be "
                                             "same as number of maps")
                    npix = 12 * data.nside ** 2
                    data.pol_iter = False
                    cols = []
                    i = 0
                    if sig is not None: 
                        j = 0
                    for map in data:
                        if sig is None:
                            cols.append(pyfits.Column(name=names[i], 
                                                    format='1E', array=map))
                        else:
                            if (i + 1) in sig:
                                cols.append(pyfits.Column(name=names[j], 
                                                    format='1E', array=map))
                                j +=1
                        i += 1
                    if data.pol_axis is None:
                        pol = False
                    else:
                        pol = True
                    cols = pyfits.ColDefs(cols)
                    prihdu = pyfits.PrimaryHDU()
                    tbhdu = pyfits.new_table(cols)
                    thdr = tbhdu.header
                    thdr.update('NSIDE', data.nside, 
                                'Resolution parameter for HEALPIX')
                    thdr.update('ORDERING', data.ordering.upper(), 
                                'Pixel ordering scheme, either RING or NESTED')
                    thdr.update('POLAR', pol, 
                                'Polarisation included (True/False)')
                    thdr.update('EXTNAME', 'HEALPIX MAP', 'HEALPix map')
                    if bad_data is None:
                        bad_data = _HPIX_BAD_DATA
                    thdr.update('BAD_DATA', bad_data, 
                                'Sentinel value given to bad pixels')
                    thdr.update('BLANK', bad_data, 
                                'Sentinel value given to bad pixels')
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
        elif isinstance(data, beammod.BeamData):
            if bintab:
                raise NotImplementedError()
            else:
                if data.pol_axis is None or data.beam.shape[data.pol_axis] == 1:
                    pol = False
                else:
                    pol = True
                if pol:
                    cols = []
                    for i in range(3):
                        if i == 0:
                            name = 'T'
                        elif i == 1:
                            name = 'E'
                        elif i == 2:
                            name = 'B'
                        if data.beam_axis == 0:
                            array = data.beam[:, i]
                        elif data.beam_axis == 1:
                            array = data.beam[i]
                        cols.append(pyfits.Column(name=name, format='E24.15', array=array))
                    cols = pyfits.ColDefs(cols, tbtype="TableHDU")
                    prihdu = pyfits.PrimaryHDU()
                    tbhdu = pyfits.new_table(cols, tbtype='TableHDU')
                    thdr = tbhdu.header
                    thdr.update('EXTNAME', 'BEAM', 
                                'Beam : b(l)')
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
