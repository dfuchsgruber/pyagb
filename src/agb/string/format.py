# Formats a string to fit into boxes

from warnings import warn

def fit_box(sequence, width, height, coder, delimiters=(0x0,), 
    scroll=0xFA, paragraph=0xFB, newline=0xFE, buffers=(0xFD,),
    control_codes={0xFC : {}}, max_buffer_size=10):
    """
    Parameters:
    -----------
    sequence : list
        The sequence to fit.
    width : int
        The number of characters that fit in a line.
    height : int
        The number of lines to fit in the box.
    coder : agb.string.agbstring.Agbstring
        Encoder / decoder for strings (only to report exceptions).
    delimiters : iterable
        Characters that can be replaced by linebreaks in order to split words.
    scroll : int
        The character to scroll a line.
    paragraph : int
        The character to clear the box.
    newline : int
        The character to force a line break.
    buffers : iterable
        The characters to access a string buffer.
    control_codes : iterable
        Control codes that skip 1 or two bytes.
    max_buffer_size : int
        The maximal number of characters that a buffer can hold.

    Returns:
    --------
    fit_sequence : list
        The sequence fit into the box
    """
    fit_sequence = []
    current_line_length = 0
    num_lines_displayed = 1
    consumed = 0 # How many following characters were already consumed

    for i, character in enumerate(sequence):
        # Take characters until the box overflows
        fit_sequence.append(character)
        if consumed > 0: # Character was already consumed before
            consumed -= 1
            continue
        if character == scroll:
            current_line_length = 0 # The number of lines does not change
        elif character == paragraph:
            current_line_length = 0
            num_lines_displayed = 1
        elif character == newline:
            if num_lines_displayed >= height:
                warn(f'Forcing a box overflow with linebreaks at \'{coder.hex_to_str(sequence[i-current_line_length:i+1] + [0xFF], 0)[0]}\'')
            current_line_length = 0
            num_lines_displayed += 1
        elif character in buffers:
            current_line_length += max_buffer_size
        elif character in control_codes:
            commands = control_codes[character]
            command = sequence[i + 1]
            consumed = 1
            arglen = commands.get(command, 0)
            # print(f'Consumed control code command {hex(command)}, Rest of string is {sequence[i + 1 + arglen :]}')
            consumed += arglen
        else:
            current_line_length += 1
        
        # Check if the line overflows
        if current_line_length > width:
            # Find the last delimiter to force a line break
            for offset in range(0, min(width + 1, len(fit_sequence))):
                if fit_sequence[-offset] in delimiters:
                    break
            
            if offset >= min(width, len(fit_sequence)):
                # The word is longer than a line
                raise RuntimeError(f'Could not force a linebreak because a single word exceeds the maximal line width at \'{coder.hex_to_str(sequence[i-offset:i+1] + [0xFF], 0)[0]}\'.')
            
            # Replace the delimiter with a linebreak or scroll
            if num_lines_displayed < height:
                fit_sequence[-offset] = newline
                num_lines_displayed += 1
            else:
                fit_sequence[-offset] = scroll
            current_line_length = offset - 1

    return fit_sequence 

        
        
