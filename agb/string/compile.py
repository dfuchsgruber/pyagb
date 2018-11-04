import os
from . import agbstring, format

# Compile a string plainly into bytes 
# Example:
# .string "..."
DIRECTIVE_STD = ".string"

# Compile a string and break it automatically to fit a box
# Example:
# .autostring WIDTH HEIGHT "..."
DIRECTIVE_AUTO = ".autostring"

# Compile a string and pad it to a certain length in bytes
# Example
# .stringpad SIZE "..."
DIRECTIVE_PADDED = ".stringpad"

def process_string(string, coder):
    """ Extracts and encodes a string embedded into '' .
    
    Parameters:
    -----------
    string : str
        The string to process.
    
    Returns:
    --------
    encoded : list
        The encoded string.
    """
    if string[0] != '\"' or string[-1] != '\"':
        raise RuntimeError(f'Expected string {string} to be embedded into \"\"')
    return coder.str_to_hex(string[1:-1])

def preprocess_assembly_line(line, coder, delimiters=(0x0,), 
    scroll=0xFA, paragraph=0xFB, newline=0xFE, buffers=(0xFD, 0xFC),
    max_buffer_size=10):
    """ Preprocesses an assembly line. 
    
    Paramters:
    ----------
    line : str
        The line to preprocess.
    coder : agbstring.Agbstring
        The encoder / decoder for strings.
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
    max_buffer_size : int
        The maximal number of characters that a buffer can hold.

    Returns:
    --------
    preprocessed : str
        The preprocessed line.
    """
    tokens = line.split()
    if len(tokens) == 0:
        return line
    # Check for directives
    if tokens[0] == DIRECTIVE_STD:
        string = line[line.index(tokens[1]):] # Keep multiple spaces
        sequence = process_string(string, coder)
    elif tokens[0] == DIRECTIVE_AUTO:
        string = line[line.index(tokens[3]):] # Keep multiple spaces
        width = int(tokens[1], base=0)
        height = int(tokens[2], base=0)
        sequence = format.fit_box(process_string(string, coder), width, height, delimiters=delimiters, 
            scroll=scroll, paragraph=paragraph, newline=newline, buffers=buffers, max_buffer_size=max_buffer_size, coder=coder)
    elif tokens[0] == DIRECTIVE_PADDED:
        string = line[line.index(tokens[2]):] # Keep multiple spaces
        size = int(tokens[1], 0)
        sequence = process_string(string, coder)
        if len(sequence) > size:
            warn(f'Sequence of size {len(sequence)} extends maximal size of {size}. Truncating.')
        else:
            sequence += [0] * (size - len(sequence))
    joined = ', '.join(map(str, sequence))
    return f'.byte {joined}'

def preprocess_assembly(input, charmap, output, tail=[0xFF], delimiters=(0x0,), 
    scroll=0xFA, paragraph=0xFB, newline=0xFE, buffers=(0xFD, 0xFC),
    max_buffer_size=10):
    """ Preprocesses an assembly file. 
    
    Parameters:
    -----------
    input : str
        The input to preprocess.
    charmap : str
        Path to the character map.
    output : str
        Path to the output file.
    tail : list
        Bytes that terminate a string.
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
    max_buffer_size : int
        The maximal number of characters that a buffer can hold.
    """
    coder = agbstring.Agbstring(charmap, tail=tail)
    with open(input, 'r', encoding='utf-8') as f:
        assembly = f.read()

    # Preprocess assembly linewise
    processed = [preprocess_assembly_line(line, coder) for line in split_continued_lines(assembly)]

    with open(output, "w+", encoding="utf-8") as f:
        f.write(os.linesep.join(processed))

        
def split_continued_lines(input):
    """ Splits an input into its lines considering line contiuations \.
    
    Parameters:
    -----------
    input : str
        The input to split
    
    Yields:
    -------
    lines : str
        The lines.
    """
    for line in input.splitlines():
        line = line.rstrip(os.linesep)
        while line.endswith('\\'):
            line = line[:-1] + next(input).rstrip(os.linesep)
        yield line

def preprocess_cpp(input, charmap, outfile, macro, tail=[0xFF]):
    """ Preprocesses a C(++) file.
    
    Parameters:
    -----------
    input : str
        The input to preprocess.
    charmap : str
        Path to the character map.
    outfile : str
        Path to the output file.
    macro : str
        The macro that encloses all strings.
    tail : list
        Bytes that terminate a string.
    """
    coder = agbstring.Agbstring(charmap, tail=tail)
    with open(input, 'r', encoding='utf-8') as f:
        source = f.read()
    sementically_irrelevant_characters = ('\t', ' ', '\\', os.linesep)

    offset = 0
    output = ''
    while offset < len(source):
        # Find occurences of the macro
        macro_offset = source.find(macro, offset)
        if macro_offset == -1:
            # No more macros to consider
            output += source[offset:]
            break
        else:
            # Found occurence of macro
            output += source[offset:macro_offset]
            offset = macro_offset + len(macro)
            output += macro

            string = ''

            num_skipped_chars = skip_sementically_irrelevant_characters(source, offset, sementically_irrelevant_characters)
            if offset + num_skipped_chars >= len(source):
                raise Exception(f'Unexpected eof while parsing macro at {source[offset:]}')
            output += source[offset:offset + num_skipped_chars]
            offset += num_skipped_chars
            if source[offset] != '(':
                raise Exception(f'Expected \'(\' after macro definition, got {source[offset]}.')
            offset += 1
            output += '('

            num_skipped_chars = skip_sementically_irrelevant_characters(source, offset, sementically_irrelevant_characters)
            if offset + num_skipped_chars >= len(source):
                raise Exception(f'Unexpected eof while expecting quotes (start of string) at {source[offset:]}')
            offset += num_skipped_chars
            if source[offset] != '"':
                raise Exception(f'Expected quotes (start of string) but got {source[offset]}')
            offset += 1
            
            # Collect characters until the string is closed
            while offset < len(source):
                offset += 1
                if source[offset - 1] == '"':
                    break
                else:
                    string += source[offset - 1]
            num_skipped_chars = skip_sementically_irrelevant_characters(source, offset, sementically_irrelevant_characters)
            if offset + num_skipped_chars >= len(source):
                raise Exception(f'Unexpected eof while expecting quotes (start of string)')
            offset += num_skipped_chars
            if source[offset] == ')':
                    # No more "..." to parse
                    offset += 1
            else:
                raise Exception(f'Expected quotes (end of string) but got {source[offset]}')
            output += '{' + ','.join(map(str, coder.str_to_hex(string))) + '})'
                
    with open(outfile, 'w+', encoding='utf-8') as f:
        f.write(output)

def skip_sementically_irrelevant_characters(string, offset, sementically_irrelevant_characters):
    """ Gets the amount of sementically irrelevant characters in a string which can be omitted for C(++) parsing.
    
    Parameters:
    -----------
    string : str
        The string in which to look for skippable characters.
    offset : int
        The current position in the string.
    sementically_irrelevant_characters : iterable
        Characters that are sementically irrelevant.
    
    Returns:
    --------
    number_of_chars : int
        The number of sementically irrelevant characters at the given offset.
    """
    for number_of_chars in range(0, len(string) - offset):
        if string[offset + number_of_chars] not in sementically_irrelevant_characters:
            break
    return number_of_chars
