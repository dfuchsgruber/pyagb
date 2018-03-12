#!/usr/bin python3

class Agbrom:
    """
    This module provides a basic interface for Gameboy Advance ROMs.
    """
    def __init__(self, path):
        fd = open(path, "rb")
        self.bytes = bytearray(fd.read())
        fd.close()
        self.path = path

    def u8(self, off):
        """ Returns an unsinged char at a given offset. """
        return int(self.bytes[off]) & 0xFFF

    def u16(self, off):
        """ Returns an unsinged short at a given offset. """
        return (int(self.bytes[off]) & 0xFF) | (int(self.bytes[off + 1]) << 8)

    def u32(self, off):
        """ Returns an unsinged integer at a given offset. """
        return (int(self.bytes[off]) & 0xFF) | (int(self.bytes[off + 1]) << 8) | (int(self.bytes[off + 2]) << 16) | (int(self.bytes[off + 3]) << 24)

    def s8(self, off):
        """ Returns a signed char at a given offset. """
        value = self.u8(off)
        return (value & 0x7F) - (value & 0x80)

    def s16(self, off):
        """ Returns a signed short at a given offset. """
        value = self.u16(off)
        return (value & 0x7FFF)  - (value & 0x8000)

    def _int(self, off):
        """ Returns a signed integer at a given offset. """
        value = self.u32(off)
        return (value & 0x7FFFFFFF) - (value & 0x80000000)

    def pointer(self, off):
        """ Interprets the unsinged integer value at off as pointer and returns the target offset. """
        return self.u32(off) - 0x8000000

    def nullable_pointer(self, off):
        """ Interprets the unsinged integer value at off as pointer and returns either the target offset
        or None if the pointer value was 0. """
        u32 = self.u32(off)
        if u32 == 0: return None
        return u32 - 0x8000000

    def array(self, off, size):
        """ Returns a list of unsigned chars that are stored at a given offset. """
        return [self.u8(off + i) for i in range (0, size)]

    def findall(self, bytes, alignment=2):
        """ Finds all occurences of a byte pattern that is properly aligned (i.e. the offset
        is divisible by 2^alignment) and returns the corresponding offsets as list."""
        modulus = 1 << alignment
        results = []
        position = -1
        bytes = bytearray(bytes)
        while True:
            position = self.bytes.find(bytes, position+1)
            if position >= 0:
                if position % modulus == 0:
                    results.append(position)
                else:
                    print("Warning. Found unaligned reference at "+hex(position))
                    
            else:
                break
        return results
    
    def get_references(self, offset):
        """ Finds all references to an offset. """
        offset += 0x08000000
        bytes = [
            offset & 0xFF,
            (offset >> 8) & 0xFF,
            (offset >> 16) & 0xFF,
            (offset >> 24) & 0xFF
        ]
        return self.findall(bytes)

    def get_repoint_patch(self, offset, label):
        """ Uses get_references to find all references to an offset and returns a patch file
        that replaces those references with a label (convenient repointing) """
        refs = self.get_references(offset)
        return "\n".join([".org " + hex(offset + 0x8000000) + "\n\t.word " + label + "\n" for offset in refs])
