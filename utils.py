from __future__ import division
import numpy as np
import lib
import mapmod
import almmod

def alm2map(ad, nside):
    mshape = list(ad.alms.shape)
    mshape[ad.ind_axis] = 12*nside**2
    md = mapmod.MapData(nside, map=np.zeros(mshape), pol_axis=ad.pol_axis)
    convalm = np.zeros((1, ad.lmax + 1, ad.mmax + 1), dtype=np.complex)
    for (map, alm) in zip(md, ad):
        for l in range(ad.lmax + 1):
            for m in range(ad.mmax + 1):
                convalm[0, l, m] = alm[almmod.lm2ind([l, m])]
        lib.alm2map_sc_d(nside, ad.lmax, ad.mmax, convalm, map[:])
    return md

def map2alm(md, lmax, mmax, weights):
    mshape = list(md.map.shape)
    mshape[md.pix_axis] = lmax * (lmax + 1) // 2 + mmax + 1
    ad = almmod.AlmData(lmax, mmax=mmax, alms = np.zeros(mshape, 
                        dtype=np.complex), pol_axis=md.pol_axis)
    convalm = np.zeros((1, ad.lmax + 1, ad.mmax + 1), dtype=np.complex)
    for (map, alm) in zip(md, ad):
        lib.map2alm_sc_d(md.nside, lmax, mmax, convalm, map, weights)
        for l in range(ad.lmax + 1):
            for m in range(ad.mmax + 1):
                alm[almmod.lm2ind([l, m])] = convalm[0, l, m]
    return ad
