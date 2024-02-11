"""A module for working with GBA roms."""

import struct
from typing import Iterable

from typing_extensions import Buffer, SupportsIndex


def find(bytes: bytearray, pattern: Iterable[SupportsIndex] | Buffer | SupportsIndex,
         alignment: int=0) -> list[int]:
    """Finds all occurences of a pattern in the rom.

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
    positions: list[int] = []
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

def references(bytes: bytearray, offset: int, alignment: int=2,
               rom_start: int =0x08000000) -> list[int]:
    """Finds all references to a given offset in the rom.

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
    references : int
        The location
    """
    pattern = bytearray(struct.pack('<I', offset + rom_start))
    return find(bytes, pattern, alignment=alignment)

