from __future__ import division
import numpy as np
import lib
import mapmod
import almmod

def alm2map(ad, nside):
    mshape = list(ad.alms.shape)
    mshape[ad.ind_axis] = 12*nside**2
    md = mapmod.MapData(nside, map=np.zeros(mshape), pol_axis=ad.pol_axis)
    ndims = len(mshape) - 1
    indlist = np.array(mshape[:md.pix_axis] + mshape[md.pix_axis + 1:])
    currind = list(np.zeros(ndims, dtype=int))
    convalms = np.zeros((1, ad.lmax + 1, ad.mmax + 1), dtype=np.complex)
    totnmaps = 1
    for el in indlist:
        totnmaps *= el
    for i in range(totnmaps):
        curralms = ad.alms[(currind[:ad.ind_axis] + [Ellipsis,] +
                            currind[ad.ind_axis:])]
        for l in range(ad.lmax + 1):
            for m in range(ad.mmax + 1):
                convalms[0, l, m] = curralms[almmod.lm2ind([l, m])]
        lib.alm2map_sc_d(nside, ad.lmax, ad.mmax, convalms, 
                    md.map[currind[:md.pix_axis] 
                + [Ellipsis,] + currind[md.pix_axis:]]) 
        traceind = ndims - 1
        while indlist[traceind] == currind[traceind] and traceind != 0:
            currind[traceind] = 0
            traceind -= 1
        currind[traceind] += 1

    return md
