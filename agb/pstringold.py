""" This module provides functionaility to convert a string in
any ASCII-like encoding to binary and vice-versa. """

class PString:

    def __init__(self, table, terminator = 0xFF):
    """ Class to convert ASCII-like strings into binary
    and vice versa.
    
    Args:
        table (str): The path the the table file. Each line
        describes a mapping sequence->byte with sequence being
        an aribtrary string and byte being in range 0...255.
        The format is seq=value."""

        self.terminator = terminator
        self.load_table(table)

    def load_table(self, table):
        """ Loads a new table file with lines describing the
        mappings sequence->byte. Each line therefore follows
        the format seq=value with seq being an arbitrary string
        and value being in range 0...255."""
        with open(table, 'r') as f:
            table = f.read()

        self.table = {}
        self.rev_table = [None] * 256

        for line in table.split("\n"):
            tokens = line.split(" ")
            if len(tokens) == 3 and tokens[1] == "=":
                #whitespace is the delimiter, but can also be a literal
                if tokens[0] == "": tokens[0] = " "
                value = int(tokens[2], 0)
                if value > 255: raise ("Non byte value associated with literal " + tokens[0])
                self.table[tokens[0]] = value
                if not self.rev_table[value]: self.rev_table[value] = tokens[0]
                    
            
    def str2hex(self, string):
        """ Parses a string and returns the binary representation. """
        string = string[:] #Create a copy of the mutable string
        bytes = []
        while len(string):
            matched_literal = None
            for literal in self.table:
                if string.find(literal) == 0:
                    matched_literal = literal
                    break
            if not matched_literal: raise ("Could not parse first literal in " + string)
            bytes.append(self.table[matched_literal])
            string = string[len(matched_literal):]
        #Append the string terminator
        bytes.append(self.terminator)
        return bytes
    
    def hex2str(self, rom, offset, decap=False):
        """ Parses binary data of  """
        result = ""
        while True:
            value = rom.u8(offset)
            if value == self.terminator: break
            if not self.rev_table[value]: raise Exception("No literal associated with "+hex(value))
            result += self.rev_table[value]
            offset += 1
        if decap:
            return decap_by_delimiters(result)
        else:
            return result

def recap_by_delimiter(s, delim):
    """ Splits a string by a delimiter, recapitalizes tokens and joins it back together"""
    return delim.join([token[0].capitalize() + token[1:] for token in s.split(delim)])

def decap_by_delimiters(s, delims=[" ", "_", "-"]):
    """ Splits a string by different delimiters, decapitalizes and joins it back together"""
    s = s.lower() #Decap everything at first
    for delim in delims:
        #print("splitting", s, "by", delim)
        s = recap_by_delimiter(s, delim)
        #print("done:",s)
    return s

def cap(s):
    """ Capitalizes all chars """
    return "".join(map(lambda c: c.capitalize(), s))



    