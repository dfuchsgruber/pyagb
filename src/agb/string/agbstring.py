"""Encoding and decoding of strings with custom mappings."""

from functools import partial
from pathlib import Path

from .trie import PrefixTriemap


class Agbstring:
    """Encoder / decoder class for strings with custom mappings."""

    def __init__(self, charmap: str, tail: tuple[int]=(0xFF,)):
        """Initializes the encoder / decoder.

        Parameters:
        -----------
        charmap : string
            Path to the mapping from strings to bytes.
        tail : int
            The byte the sequence is terminated with.
        """
        self.tail = tail
        self.str_to_hex_map: PrefixTriemap[str, tuple[int, ...]] = PrefixTriemap()
        self.hex_to_str_map: PrefixTriemap[int, str] = PrefixTriemap()

        # Parse character map
        with open(Path(charmap), 'r', encoding='utf-8') as f:
            charmap = f.read()

        for line in map(
            lambda line: line.strip().split('@')[0].strip(), # Remove comments from line
            reversed(charmap.splitlines())
            ):
            if not len(line):
                continue
            tokens = line.split('=')
            pattern, sequence = '='.join(tokens[:-1]).strip(), \
                tuple(map(partial(int, base=16),
                          filter(len, tokens[-1].split(' '))))

            # Patterns are allowed to be embedded in '' or ""
            if (pattern.startswith('\'') and pattern.endswith('\'')) or \
            (pattern.startswith('"') and pattern.endswith('"')):
                pattern = pattern[1:-1]

            self.str_to_hex_map[pattern] = sequence
            self.hex_to_str_map[sequence] = pattern

        # Insert empty strings as terminators
        self.str_to_hex_map[''] = tail
        self.hex_to_str_map[tail] = ''

    def hex_to_str(self, rom: bytearray, offset: int) -> tuple[str, int]:
        """Retrieves a string in a rom located at an offset.

        Parameters:
        -----------
        rom : bytearray
            The agbrom instance to retrieve data from.
        offset : int
            The offset of the string in the rom.

        Returns:
        --------
        string : str
            The string located at the offset.
        size : int
            The size of the bytes representation of the string.
        """
        string = ''
        size = 0
        while True:
            pattern, pattern_size = self.hex_to_str_map[rom[offset + size: offset + \
                                                            size + \
                                                            self.hex_to_str_map.max_depth]]
            if pattern_size == 0:
                raise RuntimeError(f'Unable to decrypt string at {offset + size}')
            assert pattern is not None, f'Pattern is None at {offset + size}'
            string += pattern
            size += pattern_size
            if pattern == '':
                break
        return string, size

    def str_to_hex(self, pattern: str) -> tuple[int, ...]:
        """Enocodes a string.

        Parameters:
        -----------
        pattern : str
            The string to encode.

        Returns:
        --------
        encoded : list
            A sequence of byte values representing the encoding of the string.
        """
        encoded: tuple[int, ...] = tuple()
        while True:
            sequence, size = self.str_to_hex_map[pattern]
            assert sequence is not None, f'Sequence is None at {pattern}'
            encoded += tuple(sequence)
            if pattern == '':
                break
            if size == 0:
                raise RuntimeError(f'Unable to encode {pattern}')
            pattern = pattern[size:]
        return encoded





