from __future__ import division
import numpy as np
import lib
import mapmod
import almmod
import fileutils
import psht
import subprocess
import shlex

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

def plot(md, signal='all'):
    """Uses map2gif to plot a MapData map"""
    subprocess.call(shlex.split("rm zokozokomap.fits"))
    subprocess.call(shlex.split("rm zokozokomap.gif"))
    subprocess.call(shlex.split("rm zokozokomap2.gif"))
    subprocess.call(shlex.split("rm zokozokomap3.gif"))
    fileutils.write_file('zokozokomap.fits', md)
    if signal == 'all':
        subprocess.call(shlex.split("map2gif -inp zokozokomap.fits -out zokozokomap.gif -bar true"))
        subprocess.call(shlex.split("map2gif -inp zokozokomap.fits -out zokozokomap2.gif -bar true -sig 2"))
        subprocess.call(shlex.split("map2gif -inp zokozokomap.fits -out zokozokomap3.gif -bar true -sig 3"))
        subprocess.call(shlex.split("eog zokozokomap.gif &"))

def map2gif(md, signal='all', prefix='testmap'):
    subprocess.call(shlex.split("rm " + prefix + '.fits'))
    subprocess.call(shlex.split("rm " + prefix + '.gif'))
    subprocess.call(shlex.split("rm " + prefix + '2.gif'))
    subprocess.call(shlex.split("rm " + prefix + '3.gif'))
    fileutils.write_file(prefix + '.fits', md)
    if signal == 'all':
        subprocess.call(shlex.split("map2gif -inp " + prefix + ".fits -out " + prefix + ".gif -bar true"))
        subprocess.call(shlex.split("map2gif -inp " + prefix + ".fits -out " + prefix + "2.gif -bar true -sig 2"))
        subprocess.call(shlex.split("map2gif -inp " + prefix + ".fits -out " + prefix + "3.gif -bar true -sig 3"))


def draw_gaussian_map(nside):
    npix = 12 * nside ** 2
