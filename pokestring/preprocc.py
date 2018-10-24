import os
from pokestring import pstring



def preprocess_c(filepath, charmappath, outpath, macro, terminator=0xFF):
    """ Preprocesses a c or cpp file. """

    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()
    
    offset = 0
    output = ""
    skipable_characters = set(("\t", " ", "\\", os.linesep))

    ps = pstring.Pstring(charmappath, terminator=terminator)

    while offset < len(src):
        # Parse the entire file

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

            output += src[offset:next]
            offset = next + len(macro)
            output += macro

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
    
            # Parse string
            
            
            

            # Collect characters
            string = ""

            while True:
                # Collect as many "..." sets as possible
                
                # Parse '"'
                skipped = skip(src, offset, skipable_characters)
                if offset + skipped >= len(src):
                    raise Exception("Unexpected eof while expecting \
                    '\"'.")
                
                offset += skipped

                # If got a ')' instead end parsing
                if src[offset] == ")":
                    
                    # Parse collected string and convert
                    # into bytes
                    output += "{" + ", ".join(map(
                        str, ps.str2hex(string)
                    )) + "}"
                    output += ")"
                    offset += 1
                    break
                
                if src[offset] != "\"":
                                raise Exception("Expected '\"'. Got {0}".format(
                                    src[offset:offset+10]
                                ))
                offset += 1

                # Parse characters until next '"'
                while offset < len(src):
                    if src[offset] == "\"":
                        offset += 1
                        break
                    string += src[offset]
                    offset += 1

                if offset >= len(src):
                    raise Exception("Unexpected eof while parsing \
                    string.")


            
    # Output
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
