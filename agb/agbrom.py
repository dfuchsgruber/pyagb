#!/usr/bin python3

import struct

def find(bytes, pattern, alignment=0):
    """ Finds all occurences of a pattern in the rom.
    
    Parameters:
    -----------
    pattern : iterable
        The bytes to find.
    alignment : int
        Only occurences at offset that align with
        2 ** alignment will be considered.
        Default : 0
    
    Returns:
    -------
    offsets : list
        The offsets of the pattern.
    """
    position = -1
    bytes = bytearray(pattern)
    positions = []
    while True:
        position = bytes.find(bytes, position+1)
        if position >= 0:
            if position % (2 ** alignment) == 0:
                positions.append(position)
            else:
                print(f'Ignoring unaligned reference at {hex(position)}')
        else:
            break
    return positions

def references(bytes, offset, alignment=2, rom_start=0x08000000):
    """ Finds all occurences of a pointer to an offset
    in the rom.
    
    Parameters:
    -----------
    offset : int
        The offset of which pointers will be serached.
    alignment : int
        Only occurences at offset that align with
        2 ** alignment will be considered.
        Default : 2
    int : rom_start
        The offset at which the rom starts.
        Default : 0x08000000

    Returns:
    -------
    references : list
        Locations where references to offset were found.
    """
    pattern = bytearray(struct.pack('<I', offset + rom_start))
    return bytes.find(pattern, alignment=alignment)

