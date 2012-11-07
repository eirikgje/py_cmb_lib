from __future__ import division
import numpy as np
import lib
import mapmod
import almmod
import fileutils
import psht
import subprocess
import shlex
import tempfile

def getslice(ar, axis, ind):
    """VERY handy utility routine to return a slice.

    Returns the ind'th slice along axis of array ar. Axis and ind can be lists
    or similar, in which case the nth element of axis corresponds to the nth 
    element of ind"""

    shape = ar.shape
    if isinstance(axis, int):
        sl = (slice(None),) * axis + (ind,) + (Ellipsis,)
    elif len(axis) == 2:
        if axis[0] < axis[1]:
            sl = (slice(None),) * axis[0] + (ind[0],) + (slice(None),) * \
                    (axis[1] - axis[0] - 1) + (ind[1],) + (Ellipsis,)
        else:
            sl = (slice(None),) * axis[1] + (ind[1],) + (slice(None),) * \
                    (axis[0] - axis[1] - 1) + (ind[0],) + (Ellipsis,)
    else:
        raise NotImplementedError
    return sl



def alm2map(ad, nside):
    """Determines (from whether pol_axis is set or not) whether or not to use
    polarization if polarization=True. If polarization=False, treats each alm
    as an independent alm.
    """
    if ad.ordering == 'l-major':
        ad.switchordering()
    computer = psht.PshtMmajorHealpix(lmax=ad.lmax, nside=nside, alm=ad.alms,
                                      alm_polarization=ad.pol_axis, 
                                      alm_axis=ad.ind_axis, 
                                      map_axis=ad.ind_axis,
                                      map_polarization=ad.pol_axis)
    map = computer.alm2map()
    md = mapmod.MapData(nside, map=map, pol_axis=ad.pol_axis, 
                        pol_iter = ad.pol_iter, ordering='ring')
    return md

def map2alm(md, lmax, mmax=None, weights=None):
    """Determines (from whether pol_axis is set or not) whether or not to use
    polarization if polarization=True. If polarization=False, treats each map
    as an independent map.
    """
    if mmax is None:
        mmax = lmax
    if weights is None:
        #Try to find file based on data in md
        weights = 'weight_ring_n%05d.fits' % md.nside
    if isinstance(weights, str):
        weights = fileutils.read_file(weights)
    elif not isinstance(weights, np.ndarray):
        raise TypeError("Weights must be either filename or numpy array")
    if weights.shape != (3, 2*md.nside):
        raise ValueError("Weights do not have the right shape")
    computer = psht.PshtMmajorHealpix(nside=md.nside, lmax=lmax, mmax=mmax,
                                      map=md.map,
                                      alm_polarization=md.pol_axis, 
                                      alm_axis=md.pix_axis, 
                                      map_axis=md.pix_axis,
                                      map_polarization=md.pol_axis,
                                      weights=weights[0])
    alm = computer.map2alm()
    ad = almmod.AlmData(lmax, mmax=mmax, alms=alm, pol_axis=md.pol_axis, 
                        pol_iter=md.pol_iter, ordering='m-major')
    return ad

def alm2ps(ad):
    if ad.pol_axis is not None:
        if ad.pol_axis < ad.ind_axis:
            shape = list(ad.alms.shape[:ad.pol_axis] + (6,) + ad.alms.shape[ad.pol_axis+1:ad.ind_axis] + (ad.lmax + 1) + ad.alms.shape[ad.ind_axis + 1:])
        else:
            shape = list(ad.alms.shape[:ad.ind_axis] + (ad.lmax + 1,) + ad.alms.shape[ad.ind_axis+1:ad.pol_axis] + (6,) + ad.alms.shape[ad.pol_axis + 1:])
        cd = almmod.ClData(ad.lmax, cls = np.zeros(shape), spec_axis=ad.pol_axis, spectra='all')
    else:
        shape = list(ad.alms.shape[:ad.ind_axis] + (ad.lmax + 1,) + ad.alms.shape[ad.ind_axis + 1:])
        cd = almmod.ClData(ad.lmax, cls=np.zeros(shape))
    if cd.spectra != ['TT']:
        raise NotImplementedError
    if cd.spectra == ['TT']:
        if ad.ordering == 'l-major':
            for l in range(ad.lmax + 1):
                sl = getslice(cd.cls, cd.cl_axis, l)
                ind1 = almmod.lm2ind((l, 0), lmmax=(ad.lmax, ad.mmax), \
                        ordering=ad.ordering)
                ind2 = almmod.lm2ind((l, min(l, ad.mmax)), \
                        lmmax=(ad.lmax, ad.mmax), ordering=ad.ordering)
                asl = list(getslice(ad.alms, ad.ind_axis, ind1))
                cd.cls[sl] += ad.alms[asl] ** 2
                asl[ad.ind_axis] = slice(ind1 + 1, ind2 + 1)
                cd.cls[sl] += 2 * np.sum((ad.alms[asl] * \
                        ad.alms[asl].conjugate()).real)
                cd.cls[sl] = cd.cls[sl] / (2 * l + 1)
        else:
            for l in range(ad.lmax + 1):
                sl = getslice(cd.cls, cd.cl_axis, l)
                for m in range(min(l, ad.mmax) + 1):
                    asl = getslice(ad.alms, ad.ind_axis, almmod.lm2ind((l, m), \
                            lmmax=(ad.lmax, ad.mmax), ordering=ad.ordering))
                    if m == 0:
                        cd.cls[sl] += ad.alms[asl] ** 2
                    else:
                        cd.cls[sl] += 2 * (ad.alms[asl] * \
                            ad.alms[asl].conjugate()).real
                cd.cls[sl] = cd.cls[sl] / (2 * l + 1)
    return cd

def noisemap(noise_data, nside=None):
    """Simulates a noise map.

    Now takes only diagonal noise values, but will eventually be able to
    simulate based on covariance matrices as well.

    """
    if isinstance(noise_data, mapmod.MapData):
        #Assume that the noise is diagonal, and to be multiplied by a gaussian
        gauss = np.random.standard_normal(noise_data.map.shape)
        noisemap = gauss * noise_data.map
        noise = mapmod.MapData(nside=noise_data.nside, map=noisemap, 
                                pol_axis=noise_data.pol_axis, 
                                pol_iter=noise_data.pol_iter,
                                ordering=noise_data.ordering)
    elif isinstance(noise_data, np.ndarray):
        if nside is None:
            raise ValueError("Must provide nside when noise_data is an array")
        gauss = np.random.standard_normal(noise_data.shape)
        noisemap = gauss * noise_data.map
        noise = mapmod.Mapdata(nside=nside, map=noisemap)

    return noise

def plot(md, sig=(1,), min=None, max=None, prefix=None, ncols=None, 
         common_bar=True):
    """Uses map2png to plot a MapData map"""

    if prefix is None:
        prefix = 'testmap'

    ffile = prefix + '.fits'
    pfile = prefix + '.png'
#    if common_bar or len(sig) == 1:
#        subprocess.call(shlex.split("rm " + ffile))
#        fileutils.write_file(ffile, md)
#        flags = []
#        if max is not None: flags.append('-max %f ' % max)
#        if min is not None: flags.append('-min %f ' % min)
#        for sigs in sig:
#            flags.append('-sig %2d ' % sigs)
#        if ncols is None:
#            ncols = int(np.sqrt(len(sig)))
#            flags.append('-ncol %2d' % ncols)
#        subprocess.call(shlex.split("map2png " + ffile + " " + pfile + 
#            " -bar  %s " % ''.join(flags)))
#        subprocess.call(shlex.split("eog " + pfile))
#    else:
    filelist = []
    for i in range(len(sig)):
        tffile = prefix +  '%02d.fits' % i
        tpfile = prefix + '%02d.png' % i
        filelist.append(tpfile + ' ')
        subprocess.call(shlex.split("rm " + tffile))
        fileutils.write_file(tffile, md, sig=(sig[i],))
        flags = []
        if max is not None: flags.append('-max %f ' % max[i])
        if min is not None: flags.append('-min %f ' % min[i])
        subprocess.call(shlex.split("map2png " + tffile + " " + tpfile + 
            " -bar  %s " % ''.join(flags)))

    subprocess.call(shlex.split("rm " + pfile))
    subprocess.call(shlex.split("montage -geometry +0+0 %s " % ''.join(filelist) + pfile))
    subprocess.call(shlex.split("eog " + pfile))

def map2gif(md, signal='all', prefix='testmap'):
    subprocess.call(shlex.split("rm " + prefix + '.fits'))
    subprocess.call(shlex.split("rm " + prefix + '.gif'))
    fileutils.write_file(prefix + '.fits', md)
    if signal == 'all':
        subprocess.call(shlex.split("map2gif -inp " + prefix + ".fits -out " + prefix + ".gif -bar true"))
        subprocess.call(shlex.split("map2gif -inp " + prefix + ".fits -out " + prefix + "2.gif -bar true -sig 2"))
        subprocess.call(shlex.split("map2gif -inp " + prefix + ".fits -out " + prefix + "3.gif -bar true -sig 3"))
