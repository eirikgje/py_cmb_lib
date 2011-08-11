import sys

def bin(s):
    print sys.version
    if sys.version >= '2.5':
        return str(s) if s <= 1 else bin(s >> 1) + str(s&1)  
    else:
        if s <= 1:
            return str(s)
        else:
            bin(s >> 1) + str(s&1)
