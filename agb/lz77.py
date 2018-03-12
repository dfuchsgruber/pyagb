#!/usr/bin python3

import sys

def decomp(rom, off):
    """ Decomps lz77 compressed data specified by GBATEK, returns an uint8 list"""
    #if (rom.u8(off) >> 4) & 0xF != 1: raise Exception("Lz77 Error: Data is not lz77 compressed")
    if (rom.u8(off) >> 4) & 0xF != 1: print("Lz77 Warning: Data is not lz77 compressed (at least not properly). Still trying to decompress...")
    size = rom.u8(off + 1) + (rom.u8(off + 2) << 8) + (rom.u8(off + 3) << 16) #LE read of bit 8-31 of size field
    out = [0] * size
    off += 4 #Data starts at offset = 0x4
    dst = 0
    while size:
        enc = rom.u8(off)
        off += 1
        for i in range(7, -1, -1):
            if enc & (1 << i):
                #Reference to literal chain
                ref = rom.u8(off) + (rom.u8(off+1) << 8)
                off += 2

                #Decode literal Reference
                disp = (ref >> 8) | ((ref & 0xF) << 8)
                length = ((ref >> 4) & 0xF) + 3 #Length+3 by GBATEK specs

                #Output the literal 
                for j in range(0, length):
                    try:
                        out[dst] = out[dst-disp-1]
                    except Exception as e:
                        print("Warning in lz77 decomp, malformed data :"+str(e))
                        out[dst] = 0
                    dst += 1
                    size -= 1
                    if not size: return out

            else:
                #New literal, raw dump
                out[dst] = rom.u8(off)
                off += 1
                dst += 1
                size -= 1
                if not size: return out
    return out
                
