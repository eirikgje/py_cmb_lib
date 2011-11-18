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
    subprocess.call(shlex.split("montage -geometry 1024x535 %s " % ''.join(filelist) + pfile))
    subprocess.call(shlex.split("eog " + pfile))

def map2gif(md, signal='all', prefix='testmap'):
    subprocess.call(shlex.split("rm " + prefix + '.fits'))
    subprocess.call(shlex.split("rm " + prefix + '.gif'))
    fileutils.write_file(prefix + '.fits', md)
    if signal == 'all':
        subprocess.call(shlex.split("map2gif -inp " + prefix + ".fits -out " + prefix + ".gif -bar true"))
        subprocess.call(shlex.split("map2gif -inp " + prefix + ".fits -out " + prefix + "2.gif -bar true -sig 2"))
        subprocess.call(shlex.split("map2gif -inp " + prefix + ".fits -out " + prefix + "3.gif -bar true -sig 3"))

#def map2png(md, signal='all', prefix='testmap')
