# Module to provide a encoder / decoder for strings with custom mappings.

import trie
from functools import partial

class Agbstring:
    """ Encoder / decoder class for strings with custom mappings. """

    def __init__(self, charmap, tail=0xFF):
        """ Initializes the encoder / decoder. 
        
        Parameters:
        -----------
        charmap : string
            Path to the mapping from strings to bytes.
        tail : int
            The byte the sequence is terminated with.
        """
        self.tail = tail
        self.str_to_hex = trie.PrefixTriemap()
        self.hex_to_str = trie.PrefixTriemap()

        # Parse character map
        with open(charmap, 'r', encoding='utf-8') as f:
            charmap = f.read()

        for line in map(
            lambda line: line.strip().split('@')[0].strip(), # Remove comments from line
            charmap.splitlines()
            ):
            if not len(line):
                continue
            tokens = line.split('=')
            pattern, sequence = eval(tokens[:-1]), tuple(map(partial(int, base=16), filter(len, tokens[-1].split(' '))))
            self.str_to_hex[pattern] = sequence
            self.hex_to_str[sequence] = pattern

        # Insert empty strings as terminators
        self.str_to_hex[''] = tail
        self.hex_to_str[tail] = ''
        
    def hex_to_str(self, rom, offset):
        """ Retrieves a string in a rom located at an offset.
        
        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
            The agbrom instance to retrieve data from.
        offset : int
            The offset of the string in the rom.
            
        Returns:
        --------
        string : str
            The string located at the offset.
        """
        
        string = ''
        while True:
            pattern, size = self.hex_to_str[sequence]
            




    