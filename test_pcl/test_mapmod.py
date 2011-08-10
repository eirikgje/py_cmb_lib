import mapmod
import numpy as np
from nose.tools import ok_, eq_, assert_raises

nside = 32
npix = 12*nside**2
map = np.arange(npix)

def test_ordering_conversion():
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
    md = mapmod.MapData(map=map, ordering='ring', nside=nside)
    yield ok_, np.all(md.map == map)
    yield ok_, md.nside == nside
    yield ok_, md.ordering == 'ring'

def test_init():
    def func():
        md = mapmod.MapData(map=map, nside=32.0, ordering='ring')
    yield assert_raises, TypeError, func
    #Should not be able to set nside different from map nside
    def func():
        md = mapmod.MapData(map=map, nside=12, ordering='ring')
    yield assert_raises, ValueError, func
    def func():
        md = mapmod.MapData(map=map, nside=nside, ordering='ringe')
    yield assert_raises, ValueError, func
    def func():
        md = mapmod.MapData(map=4, nside=nside, ordering='ring')
    yield assert_raises, TypeError, func

def test_assign():
    md = mapmod.MapData(map=map, nside=nside, ordering='ring')
    def func():
        md.map = 4
    yield assert_raises, TypeError, func
    def func():
        md.nside = 12.0
    yield assert_raises, TypeError, func
    def func():
        md.nside = 12
    yield assert_raises, ValueError, func
    def func():
        md.ordering = 'neste'
    yield assert_raises, ValueError, func
