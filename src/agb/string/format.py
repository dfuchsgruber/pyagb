"""Formats strings to fit into a box."""

from warnings import warn

from agb.string.agbstring import Agbstring


def fit_box(sequence: tuple[int, ...], width: int, height: int, coder: Agbstring,  # noqa: C901
               delimiters: tuple[int]=(0x0,), scroll: int=0xFA, paragraph: int=0xFB,
               newline: int=0xFE, buffers: tuple[int]=(0xFD,),
                control_codes: dict[int, dict[int, int]]={0xFC : {}},
                max_buffer_size: int=10) -> tuple[int, ...]:
    """Fits a sequence into a box.

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
    fit_sequence: list[int] = []
    current_line_length = 0
    num_lines_displayed = 1
    consumed: int = 0 # How many following characters were already consumed

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
                warn(f'Forcing a box overflow with linebreaks at \'{coder.hex_to_str(
                    bytearray(sequence[i-current_line_length:i+1] + (0xFF,)), 0)[0]}\'')
            current_line_length = 0
            num_lines_displayed += 1
        elif character in buffers:
            current_line_length += max_buffer_size
        elif character in control_codes:
            commands = control_codes[character]
            command = sequence[i + 1]
            consumed = 1
            arglen = commands.get(command, 0)
            consumed += arglen
        else:
            current_line_length += 1

        # Check if the line overflows
        if current_line_length > width:
            # Find the last delimiter to force a line break
            for offset in range(0, min(width + 1, len(fit_sequence))):
                if fit_sequence[-offset] in delimiters:
                    break
            else:
                offset = -1

            if offset >= min(width, len(fit_sequence)):
                # The word is longer than a line
                raise RuntimeError(f'Could not force a linebreak because a '
                                   'single word  exceeds the maximal line width ' \
                                    f'at \'{coder.hex_to_str(bytearray(
                                        sequence[i-offset:i+1] + (0xFF,)), 0)[0]}\'.')

            # Replace the delimiter with a linebreak or scroll
            if num_lines_displayed < height:
                fit_sequence[-offset] = newline
                num_lines_displayed += 1
            else:
                fit_sequence[-offset] = scroll
            current_line_length = offset - 1

    return tuple(fit_sequence)



