"""Lz77 compression and decompression for the agb."""

import struct
import warnings
from typing import Sequence


def decompress(data: Sequence[int], offset: int=0) -> list[int]:
    """Decompresses lz77 compressed data specified by GBATEK for the agb.

    Parameters:
    -----------
    rom : Sequence[int]
        The rom instance to obtain data from
    offset : int
        The offset of the data to decompress.

    Yields:
    -------
    data : int
        Bytes of the decompressed data
    """
    data = bytes(data)

    header = struct.unpack_from('<I', data, offset)[0]
    magic = header & 0xFF
    size = header >> 8

    if magic >> 4 != 1:
        warnings.warn(f'No proper lz77 header found at {hex(offset)}. ' \
            'Trying to decrompess nonetheless...')

    literals = [0] * size
    literal_offset = 0
    offset += 4

    while size:
        # Get the types of the 8 next factors
        encoded = header = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        for i in range(7, -1, -1):
            if size == 0:
                # No more data remaining
                break
            # Process factors (from msb to lsb)
            # print(f'Before block {7 - i}: {list(literals[:literal_offset])}')
            if encoded & (1 << i):
                # Factor refers to the literal chain
                ref = struct.unpack_from('<H', data, offset)[0]
                offset += 2

                # Decode literal Referencodede
                # Displacement w.r.t. refered frame
                displacement = (ref >> 8) | ((ref & 0xF) << 8)
                if displacement == 0:
                    warnings.warn('Lz77 data contains VRAM-unsafe compression ' \
                        '(DISP=0)...')

                # Length + 3 due to GBATEK speces (no frame is smaller)
                length = ((ref >> 4) & 0xF) + 3

                # Copy from from the literals
                for _ in range(length):
                    literals[literal_offset] = literals[literal_offset -
                                                        displacement - 1]
                    literal_offset += 1
                    size -= 1
            else:
                # Current factor is not in the literal frame
                literals[literal_offset] = struct.unpack_from('<B', data, offset)[0]
                offset += 1
                literal_offset += 1
                size -= 1
    return literals

def compress(data: Sequence[int]) -> bytearray:
    """Compresses data with lz77 compression.

    Paramters:
    ----------
    data : bytes
        The data to compress (char values).

    Returns:
    -------
    compressed : bytearray
        The bytes of the compressed sequence.
    """
    # Create the header
    data = bytes(data)
    header = 0x10 | (len(data) << 8)
    literals = bytearray(struct.pack('<i', header))

    offset = 0
    # Start compressing
    while offset < len(data):
        # print(f'Compressing at offset {hex(offset)}...')
        # Append the encoding for the next 8 factors
        encoding_offset = len(literals)
        literals.append(0)

        for i in range(7, -1, -1):
            if offset >= len(data):
                continue
            # Create a new factor
            # Only consider literals that are in 12-bit range (-1)
            # and do not overlap with the current 8-blocks as the
            # encoding byte is not fixed yet.

            # Only consider factors within 12-bit range
            window_start = max(0, offset - (1 << 12))

            match_offset, match_size = -1, -1
            # The factors have minimum size of 3
            # The factors can only have a 4-bit size, but the size is 3 based
            for factor_size in range(3, 16 + 3):
                if offset + factor_size >= len(data):
                    break
                factor = data[offset : offset + factor_size]
                # print(f'Seraching for factor {list(factor)}')
                # Do not allow displacements of 1 since in VRAM halfwords will
                # be copied. Thus displacements of 1 may induce malformed data copies
                factor_offset = data.find(factor, window_start, offset +
                                          factor_size - 2)
                if factor_offset < 0:
                    # The factor is not availible -> all bigger factors won't be as well
                    break
                match_size = factor_size
                match_offset = factor_offset

            if match_size >= 3:
                # Factor is already part of the literal chain
                literals[encoding_offset] |= 1 << i

                displacement = offset - match_offset - 1
                assert(displacement) >= 1, 'Displacement is 0'
                msb = (displacement >> 8) | ((match_size - 3) << 4)
                lsb = displacement & 0xFF
                literals.append(msb)
                literals.append(lsb)
                offset += match_size
            else:
                # print(f'Raw dump')
                literals.append(data[offset])
                offset += 1

    return literals
