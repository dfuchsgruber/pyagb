#!/usr/bin/env python3

import os
from pokestring import pstring

STATE_NONE, STATE_EXPECT, STATE_NEXT_OR_CLOSE, STATE_PARSE = range(4)


def preprocess_c(filepath, charmappath, outpath, macro, terminator=0xFF):
    """ Preprocesses a c or cpp file. """

    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()
    
    offset = 0
    output = ""
    state = STATE_NONE

    skipable_characters = set(("\t", " ", "\\", os.linesep))

    ps = pstring.Pstring(charmappath, terminator=terminator)

    while offset < len(src):
        # Parse the entire file
        if state == STATE_NONE:
            # Find the next occurence of macro
            next = src.find(macro, offset)

            if next == -1:
                # No more occrences of the macro
                # simply append the rest of the file
                output += src[offset:]
                offset = len(src)
            else:
                # Macro found, skip sementically
                # irrelevant characters

                output += src[offset:offset+next]
                offset = next + len(macro)

                skipped = skip(src, offset, skipable_characters)
                if offset + skipped >= len(src):
                    raise Exception("Unexpected eof while parsing \
                    macro.")
                output += src[offset:offset+skipped]
                offset += skipped
                
                if src[offset] != '(':
                    raise Exception("Expected '(' after macro \
                    definition.")
                
                offset += 1
                output += '('

                state = STATE_EXPECT
        
        elif state == STATE_EXPECT:
            # Expect '"'
            skipped = skip(src, offset, skipable_characters)
            if offset + skipped >= len(src):
                raise Exception("Unexpected eof while expecting \
                a list of strings to parse.")
            
            output += src[offset:offset+skipped]
            offset += skipped
            
            if src[offset] == "\"":
                # Begin of new string
                state = STATE_PARSE
                offset += 1
                string = ""
            else:
                raise Exception("Unexpected character {0}. \
                expected either '\"'".format(src[offset]))
            
        elif state == STATE_PARSE:
            # Parse characters until '"' is encountered
            if src[offset] == "\"":
                # String ends
                bytes = ps.str2hex(string)
                output += "{" + ", ".join(map(
                    str, bytes
                )) + "}"
                offset += 1
                state = STATE_NEXT_OR_CLOSE
            else:
                # Character will be collected
                string += src[offset]
                offset += 1
        
        elif state == STATE_NEXT_OR_CLOSE:
            # Expect either a comma or a ')'
            # to enclose the macro
            skipped = skip(src, offset, skipable_characters)
            if offset + skipped >= len(src):   
                raise Exception("Unexpected eof while expecting \
                either a comma or ')' for macros.")
            offset += skipped
            
            if src[offset] == ',':
                output += ','
                offset += 1
                state = STATE_EXPECT
            elif src[offset] == ')':
                output += ')'
                offset += 1
                state = STATE_NONE
            
    # Outut
    with open(outpath, "w+", encoding="utf-8") as f:
        f.write(output)


def skip(string, offset, characters):
    """ Returns the number of whitespace or linebreak characters
    that are at the offset and thus must be skipped. """
    number = 0
    while offset + number < len(string) and \
        string[offset + number] in characters:
        number += 1
    return number
