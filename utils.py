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
    currind = np.zeros(ndims)
    convalms = np.zeros((1, ad.lmax + 1, ad.mmax + 1), dtype=np.complex)
    for i in range(ndims):
        for j in range(indlist[i]):
            curralms = ad.alms[(currind[:nd.indaxis] + [Ellipsis,] +
                                currind[nd.indaxis:])]
            for l in range(ad.lmax + 1):
                for m in range(ad.mmax + 1):
                    convalms[1, l, m] = curralms[almmod.lm2ind([l, m])]
            alm2map(nside, ad.lmax, ad.mmax, convalms, 
                    md.map[currind[:nd.indaxis] + [Ellipsis,] + 
                           currind[nd.indaxis:]]) 
            currind[i] += 1
            #TODO: Fix this
