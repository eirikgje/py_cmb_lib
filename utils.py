from __future__ import division
import lib
import mapmod
import almmod

def alm2map(ad, nside):
    mshape = ad.alms.shape
    mshape[ad.indaxis] = 12*nside**2
    md = mapmod.MapData(nside, map=np.zeros(mshape), pol_axis=ad.pol_axis)
    ndims = len(mshape) - 1
    indlist = np.array(mshape[:md.indaxis] + mshape[md.indaxis + 1:])
    currind = np.zeros(indlist, dtype=int)
    for i in range(len(indlist)):
        for j in range(indlist[i]):
            curralms = ad.alms[currind[:

    #for i in range(ndims):
    #    if i != md.indaxis:
    #        for j in range(mshape[i]):
