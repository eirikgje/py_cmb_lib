import mapmod
import numpy as np
from nose.tools import ok_, eq_, assert_raises
import sys

nside = 32
npix = 12*nside**2
map = np.arange(npix)

def test_ordering_conversion():
    r2npixs = {0:15, 44:95, 112:99, 55:52}
    n2rpixs = {0:74, 106:94, 66:135, 176:191}
    nside = 4
    npix = 12*nside**2
    map = np.arange(npix)
    md = mapmod.MapData(map=map, ordering='ring', nside=nside)
    md.switchordering()
    for key, value in r2npixs.items():
        yield eq_, md.map[0, key], value
    md.switchordering()
    yield ok_, np.all(map == md.map)

    md = mapmod.MapData(map=map, ordering='nested', nside=nside)
    md.switchordering()
    for key, value in n2rpixs.items():
        yield eq_, md.map[0, key], value
    md.switchordering()
    yield ok_, np.all(map == md.map)

    #Reload the module because init_r2n and init_n2r acts differently depending
    #on whether the other has been initialized:
    reload(mapmod)
    md = mapmod.MapData(map=map, ordering='nested', nside=nside)
    md.switchordering()
    for key, value in n2rpixs.items():
        yield eq_, md.map[0, key], value
    md.switchordering()
    yield ok_, np.all(map == md.map)

    md = mapmod.MapData(map=map, ordering='ring', nside=nside)
    md.switchordering()
    for key, value in r2npixs.items():
        yield eq_, md.map[0, key], value
    md.switchordering()
    yield ok_, np.all(map == md.map)

    #Test for other nsides
    nside = 2
    npix = 12*nside**2
    map = np.arange(npix)
    md = mapmod.MapData(map=map, ordering='ring', nside=nside)
    md.switchordering()
    md.switchordering()
    yield ok_, np.all(map == md.map)

    nside = 32
    npix = 12*nside**2
    map = np.arange(npix)
    md = mapmod.MapData(map=map, ordering='ring', nside=nside)
    md.switchordering()
    md.switchordering()
    yield ok_, np.all(map == md.map)

    #Test for other shapes
    md = mapmod.MapData(nside, subd=(3, 4))
    md.switchordering()
    md.switchordering()
    yield eq_, md.map.shape, (3, 4, 1, npix)


def test_sanity():
    md = mapmod.MapData(nside=nside)
    yield ok_, md.nside == nside
    md = mapmod.MapData(nside, ordering='ring')
    yield ok_, md.ordering == 'ring'
    md = mapmod.MapData(nside, ordering='nested')
    yield ok_, md.ordering == 'nested'
    map = np.arange(npix)
    md = mapmod.MapData(nside, map=map)
    yield ok_, np.all(md.map == map)
    md = mapmod.MapData(nside, subd=(3, 2, 1))
    yield ok_, np.all((3, 2, 1) == md.subd)

def test_init():
    def func():
        md = mapmod.MapData(nside=32.0)
    yield assert_raises, TypeError, func
    #Should not be able to set nside different from map nside
    def func():
        md = mapmod.MapData(nside=12, map=map)
    yield assert_raises, ValueError, func
    def func():
        md = mapmod.MapData(nside, ordering='ringe')
    yield assert_raises, ValueError, func
    def func():
        md = mapmod.MapData(nside, map=4)
    yield assert_raises, TypeError, func
    #Given no map, should initialize a map of zeros with given nside
    md = mapmod.MapData(nside=nside)
    yield ok_, np.all(md.map == np.zeros((1, npix)))

def test_assign():
    md = mapmod.MapData(nside)
    def func():
        md.map = 4
    yield assert_raises, TypeError, func
    def func():
        md.nside = 12.0
    yield assert_raises, TypeError, func
    #TODO: Assignment of nside - up/degradation of map?
    #def func():
    #    md.nside = 12
    #yield assert_raises, ValueError, func
    def func():
        md.ordering = 'neste'
    yield assert_raises, ValueError, func

def test_shape():
    md = mapmod.MapData(nside)
    yield eq_, (1, npix), md.map.shape
    md.subdivide(5)
    yield eq_, (5, 1, npix), md.map.shape
    map = np.arange(npix)
    def func():
        md.map = map
    yield assert_raises, ValueError, func
    map.resize((2,3,4,npix))
    yield assert_raises, ValueError, func
    md = mapmod.MapData(nside, subd=(3, 2))
    yield eq_, (3, 2, 1, npix), md.map.shape
