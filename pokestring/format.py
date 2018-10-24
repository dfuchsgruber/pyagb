SPACE = 0x0
SCROLLINE = 0xFA
PARAGRAPH = 0xFB
BUFFER = 0xFD
NEWLINE = 0xFE

MAX_BUFFERSIZE = 10

def format_pstring(bytes, linewidth, line_cnt):
    """ Formats a pokestring such that it contains
    linewidth characters per line at maximum.
    It will break the lines first with '\n'
    and then use '\l'."""

    output = []
    len_line = 0 # Keep track of the current line's length
    lines = 1 # Keep track of how many lines are displayed

    for byte in bytes:

        output.append(byte)

        # Parse byte until the line overflows
        if byte == SCROLLINE:
            # Line is scrolled upwards
            len_line = 0
        elif byte == PARAGRAPH:
            # Box is cleared.
            len_line = 0
            lines = 1
        elif byte == NEWLINE:
            # Newline forced.
            if byte != NEWLINE and \
                lines >= line_cnt:
                # There is no room for
                # additional lines
                print("Warning. There is \
                no place for further lines \
                in the box and thus a forced \
                linebreak {0} is not \
                recommended".format(hex(NEWLINE)))
            len_line = 0
            lines += 1
        elif byte in (BUFFER, 0xFC):
            # Buffers and misc commands
            len_line += MAX_BUFFERSIZE
        else:
            # Casual character
            len_line += 1
        
        # Check if line overflows
        if len_line > linewidth:
            # Find last space in output
            pos = 0
            while pos < len(output):
                if output[-pos] == SPACE:
                    break
                pos += 1

            # If no space was found, raise Exception
            if pos >= len(output) or \
                output[-pos] != SPACE:
                raise Exception("Error. Could not \
                break the line because \
                a single sequence without \
                any spaces is too long \
                for the line!")
                
            len_line = pos

            # If there can be more lines
            # displayed use a NEWLINE as splitter
            # and else a SCROLLINE
            if lines < line_cnt:
                output[-pos] = NEWLINE
                lines += 1
            else:
                output[-pos] = SCROLLINE
    return output
