import __builtin__
import sys

if sys.version >= '2.5':
    any = __builtin__.any
    all = __builtin__.all
else:
    def any(arr):
        return bool(sum(arr))
        
    def all(arr):
        return len(arr) <= sum(arr)

if sys.version >= '2.6':
    bin = __builtin__.bin
else:
    def bin(s):
        if s <= 1:
            return '0b' + str(s)
        else:
            return bin(s >> 1) + str(s&1)

