from __future__ import division
import numpy as np
import lib
import mapmod
import almmod
import fileutils
import psht

def alm2map(ad, nside, polarization=True):
    """Determines (from whether pol_axis is set or not) whether or not to use
    polarization if polarization=True. If polarization=False, treats each alm
    as an independent alm.
    """
    mshape = list(ad.alms.shape)
    mshape[ad.ind_axis] = 12*nside**2
    if polarization:
        if ad.pol_axis is None:
            ad.pol_iter = False
        else:
            ad.pol_iter = True
    else:
        ad.pol_iter = False
    md = mapmod.MapData(nside, map=np.zeros(mshape), pol_axis=ad.pol_axis, 
                        pol_iter = ad.pol_iter)
    convalm = np.zeros((1, ad.lmax + 1, ad.mmax + 1), dtype=np.complex)
    if ad.ordering == 'l-major':
        ad.switchordering()
    info = psht.PshtMmajorHealpix(ad.lmax, nside, 1)
    for (map, alm) in zip(md, ad):
        if not ad.pol_iter:
            map[:] = info.alm2map(alm.reshape(1, alm.shape[0]))
        else:
            map[:] = info.alm2map(alm)
    return md

def map2alm(md, lmax, mmax=None, weights=None, polarization=True):
    """Determines (from whether pol_axis is set or not) whether or not to use
    polarization if polarization=True. If polarization=False, treats each map
    as an independent map.
    """
    if mmax is None:
        mmax = lmax
    if polarization:
        if md.pol_axis is None:
            md.pol_iter = False
        else:
            md.pol_iter = True
    else:
        md.pol_iter = False
    if md.ordering == 'nested':
        md.switchordering()
    if weights is None:
        #Try to find file based on data in md
        weights = 'weight_ring_n%05d.fits' % md.nside
    if isinstance(weights, str):
        weights = fileutils.read_file(weights)
    elif not isinstance(weights, np.ndarray):
        raise TypeError("Weights must be either filename or numpy array")
    if weights.shape != (3, 2*md.nside):
        raise ValueError("Weights do not have the right shape")
    mshape = list(md.map.shape)
    mshape[md.pix_axis] = lmax * (lmax + 1) // 2 + mmax + 1
    ad = almmod.AlmData(lmax, mmax=mmax, alms = np.zeros(mshape, 
                        dtype=np.complex), pol_axis=md.pol_axis, 
                        pol_iter=md.pol_iter, ordering='m-major')
    info = psht.PshtMmajorHealpix(ad.lmax, md.nside, 1, weights[0])
    for (map, alm) in zip(md, ad):
        if md.pol_iter:
            alm[:] = info.map2alm(map)
        else:
            alm[:] = info.map2alm(map.reshape(1, map.shape[0]))
    return ad

