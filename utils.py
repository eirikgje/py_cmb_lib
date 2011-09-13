from __future__ import division
import numpy as np
import lib
import mapmod
import almmod
import fileutils

def alm2map(ad, nside):
    mshape = list(ad.alms.shape)
    mshape[ad.ind_axis] = 12*nside**2
    md = mapmod.MapData(nside, map=np.zeros(mshape), pol_axis=ad.pol_axis)
    convalm = np.zeros((1, ad.lmax + 1, ad.mmax + 1), dtype=np.complex)
    ad.pol_iter = False
    for (map, alm) in zip(md, ad):
        for l in range(ad.lmax + 1):
            for m in range(ad.mmax + 1):
                convalm[0, l, m] = alm[almmod.lm2ind([l, m])]
        lib.alm2map_sc_d(nside, ad.lmax, ad.mmax, convalm, map[:])
    return md

def map2alm(md, lmax, mmax, weights):
    if md.ordering == 'nested':
        md.switchordering()
    if isinstance(weights, str):
        weights = fileutils.read_file(weights)
    elif not isinstance(weights, np.ndarray):
        raise TypeError("Weights must be either filename or numpy array")
    if weights.shape != (3, 2*md.nside):
        raise ValueError("Weights do not have the right shape")
    mshape = list(md.map.shape)
    mshape[md.pix_axis] = lmax * (lmax + 1) // 2 + mmax + 1
    if md.pol_axis is None:
        pol_iter = False
    else:
        pol_iter = True
    md.pol_iter = pol_iter
    ad = almmod.AlmData(lmax, mmax=mmax, alms = np.zeros(mshape, 
                        dtype=np.complex), pol_axis=md.pol_axis, 
                        pol_iter=pol_iter)
    convalm = np.zeros((1, ad.lmax + 1, ad.mmax + 1), dtype=np.complex)
    for (map, alm) in zip(md, ad):
        if pol_iter:
            for i in range(3):
                if md.pol_axis < md.pix_axis:
                    lib.map2alm_sc_d(md.nside, lmax, mmax, map[i], convalm,
                                     weights[i])
                    for l in range(ad.lmax + 1):
                        for m in range(ad.mmax + 1):
                            alm[i, almmod.lm2ind([l, m])] = convalm[0, l, m]
                else:
                    lib.map2alm_sc_d(md.nside, lmax, mmax, convalm, map[:, i],
                                     weights[i])
                    for l in range(ad.lmax + 1):
                        for m in range(ad.mmax + 1):
                            alm[almmod.lm2ind([l, m]), i] = convalm[0, l, m]
        else:
            lib.map2alm_sc_d(md.nside, lmax, mmax, convalm, map, weights[0])
            for l in range(ad.lmax + 1):
                for m in range(ad.mmax + 1):
                    alm[almmod.lm2ind([l, m])] = convalm[0, l, m]
    return ad
