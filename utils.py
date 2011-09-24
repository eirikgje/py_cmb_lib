from __future__ import division
import numpy as np
import lib
import mapmod
import almmod
import fileutils
import random

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

def shape_conversion(obj, newclass, keyval):
    if isinstance(obj, newclass)
        raise TypeError("Same Class")
    if isinstance(obj1, almmod.ClData):
        shape = obj1.cls.shape
        pri_axis = obj1.cl_axis
        sec_axis = obj1.spec_axis
    elif isinstance(obj1, almmod.AlmData):
        shape = obj1.alms.shape
        pri_axis = obj1.ind_axis
        sec_axis = obj1.pol_axis
    elif isinstance(obj1, mapmod.MapData):
        shape = obj1.map.shape
        pri_axis = obj1.pix_axis
        sec_axis = obj1.pol_axis
    if isinstance(newclass, almmod.ClData):
        shape[pri_axis] = 
        if sec_axis is not None:
            shape[sec_axis] = obj2.cls.shape[obj2.spec_axis]
    if isinstance(obj2, almmod.AlmData):
        shape[pri_axis] = obj2.alms.shape[obj2.ind_axis]
        if sec_axis is not None:
            shape[sec_axis] = obj2.alms.shape[obj2.pol_axis]
    if isinstance(obj2, mapmod.MapData):
        shape[pri_axis] = obj2.map.shape[obj2.pix_axis]
        if sec_axis is not None:
            shape[sec_axis] = obj2.map.shape[obj2.pol_axis]
    return shape

#def gen_AlmData(cd):
#    if not isinstance(cd, ClData):
#        raise TypeError("cd must be ClData object")
#    if cd.pol_axis is None:
#        return AlmData(cd.lmax, alms=np.zeros(cd.cls.shape[:cd.cl_axis] + 
#            (lmax * (lmax + 1) // 2 + lmax + 1,) + 
#            cd.cls.shape[cd.cl_axis + 1:], dtype=np.complex), ind_axis=cd.cl_axis)
#    else:
#        if cd.nspecs == 6:
#            if cd.spec_axis < cd.cl_axis:
#                return AlmData(cd.lmax, 
#                    alms=np.zeros(cd.cls.shape[:cd.spec_axis] + (3,) + 
#                    cd.cls.shape[cd.spec_axis + 1:cd.cl_axis] + 
#                    (lmax * (lmax + 1) // 2 + lmax + 1,) + 
#                    cd.cls.shape[cd.cl_axis + 1:], dtype=np.complex), ind_axis=cd.cl_axis)
#            else:
#                return AlmData(cd.lmax, 
#                    alms=np.zeros(cd.cls.shape[:cd.cl_axis] +  
#                    (lmax * (lmax + 1) // 2 + lmax + 1,) +
#                    cd.cls.shape[cd.cl_axis + 1:cd.spec_axis] + (3,) + 
#                    cd.cls.shape[cd.spec_axis + 1:], dtype=np.complex), ind_axis=cd.cl_axis)

def draw_alms(cd, beam=None):
    if beam is not None:
        raise NotImplementedError()
    if cd.spec_axis is not None:
        cd.spec_iter = True
    lmax = cd.lmax
    #alms = np.zeros((lmax * (lmax + 1) // 2 + lmax + 1, 3), dtype=np.complex)
    if cd.nspecs == 6:
        alms = np.zeros(, dtype=np.complex)
        if cd.spec_axis < cd.cl_axis:
            ad = almmod.AlmData(lmax, pol_axis=cd.spec_axis, 
                    pol_iter=cd.spec_iter, alms = alms, ind_axis=cd.cl_axis)
        else:
            alms = alms.swapaxes(cd.spec_axis, cd.cl_axis)
            print alms.shape
            ad = almmod.AlmData(lmax, pol_axis=cd.cl_axis, 
                    pol_iter=cd.spec_iter, alms=alms, ind_axis=cd.spec_axis)
    else:
        raise NotImplementedError()
    for (cls, alms) in zip(cd, ad):
        if cd.pol_iter:
            if cd.spec_axis < cd.cl_axis: 
                mat = almmod.lin2mat(cls)
            else:
                mat = almmod.lin2mat(cls.swapaxes(0, 1))
        else:
            mat = almmod.lin2mat(cls)
        chol = np.zeros(mat.shape)
        for l in range(2, lmax + 1):
            chol[:, :, l] = np.linalg.cholesky(mat[:, :, l])
        x = np.zeros((2, 3))
        onebysqrt2 = 1/np.sqrt(2)
        for l in range(2, lmax + 1):
            for m in range(l + 1):
                for i in range(2):
                    eta = np.random.standard_normal(3)
                    x[i] = np.dot(eta, chol[l])
                for i in range(3):
                    if m == 0:
                        alms[almmod.lm2ind((l, m)), i] = np.complex(x[0, i], 
                                                                      0.0)
                    else:
                        alms[almmod.lm2ind((l, m)), i] = (onebysqrt2 * 
                                                    np.complex(x[0, i], 
                                                               x[1, i]))
    return almmod.AlmData(lmax, pol_axis=1, alms=alms)
