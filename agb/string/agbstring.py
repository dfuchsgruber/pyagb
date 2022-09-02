# Module to provide a encoder / decoder for strings with custom mappings.

from pathlib import Path
from . import trie
from functools import partial

class Agbstring:
    """ Encoder / decoder class for strings with custom mappings. """

    def __init__(self, charmap, tail=[0xFF]):
        """ Initializes the encoder / decoder. 
        
        Parameters:
        -----------
        charmap : string
            Path to the mapping from strings to bytes.
        tail : int
            The byte the sequence is terminated with.
        """
        self.tail = tail
        self.str_to_hex_map = trie.PrefixTriemap()
        self.hex_to_str_map = trie.PrefixTriemap()

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
            pattern, sequence = '='.join(tokens[:-1]).strip(), tuple(map(partial(int, base=16), filter(len, tokens[-1].split(' '))))

            # Patterns are allowed to be embedded in '' or ""
            if (pattern.startswith('\'') and pattern.endswith('\'')) or (pattern.startswith('"') and pattern.endswith('"')):
                pattern = pattern[1:-1]

            self.str_to_hex_map[pattern] = sequence
            self.hex_to_str_map[sequence] = pattern

        # Insert empty strings as terminators
        self.str_to_hex_map[''] = tail
        self.hex_to_str_map[tail] = ''
        
    def hex_to_str(self, rom, offset):
        """ Retrieves a string in a rom located at an offset.
        
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
            pattern, pattern_size = self.hex_to_str_map[rom[offset + size:]]
            if pattern_size == 0:
                raise RuntimeError(f'Unable to decrypt string at {offset + size}')
            string += pattern
            size += pattern_size
            if pattern == '':
                break
        return string, size
    
    def str_to_hex(self, pattern):
        """ Enocodes a string.
        
        Parameters:
        -----------
        pattern : str
            The string to encode.
        
        Returns:
        --------
        encoded : list
            A sequence of byte values representing the encoding of the string.
        """
        encoded = []
        while True:
            sequence, size = self.str_to_hex_map[pattern]
            encoded += sequence
            if pattern == '':
                break
            if size == 0:
                raise RuntimeError(f'Unable to encode {pattern}')
            pattern = pattern[size:]
        return encoded
            




    