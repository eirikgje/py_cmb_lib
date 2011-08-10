import mapmod
import numpy as np
from nose.tools import ok_, eq_

nside = 32
npix = 12*nside**2

def test_ordering_conversion():
    map = np.arange(npix)
    r2npixs = {0:1022, 40:1006, 7654:5386, 12287:11264}
    n2rpixs = {0:5968, 100:5202, 345:2521, 10000:12188}
    md = mapmod.MapData(map=map, ordering='ring', nside=nside)
    md.switchordering()
    for key, value in r2npixs.items():
        yield eq_, md.map[0, key], value

    md = mapmod.MapData(map=map, ordering='nested', nside=nside)
    md.switchordering()
    for key, value in n2rpixs.items():
        yield eq_, md.map[0, key], value

def test_sanity():
    map = np.arange(npix)
    md = mapmod.MapData(map=map)
    yield ok_, np.all(md.map == map)
