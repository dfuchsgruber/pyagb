import os
from . import agbstring, format
from warnings import warn

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

def preprocess_assembly_line(line, project):
    """ Preprocesses an assembly line. 
    
    Paramters:
    ----------
    line : str
        The line to preprocess.
    project : pymap.project.Project
        The pymap project.

    Returns:
    --------
    preprocessed : str
        The preprocessed line.
    """
    tokens = line.split()
    if len(tokens) == 0:
        return line
    # Check for directives
    if tokens[0] == project.config['string']['as']['directives']['std']:
        string = line[line.index(tokens[1]):] # Keep multiple spaces
        sequence = process_string(string, project.coder)
    elif tokens[0] == project.config['string']['as']['directives']['auto']:
        string = line[line.index(tokens[3]):] # Keep multiple spaces
        width = int(tokens[1], base=0)
        height = int(tokens[2], base=0)
        characters = project.config['string']['characters']
        sequence = format.fit_box(process_string(string, project.coder), width, height, delimiters=characters['delimiters'], 
            scroll=characters['scroll'], paragraph=characters['paragraph'], newline=characters['newline'], buffers=characters['buffers'],
            max_buffer_size=characters['max_buffer_size'], coder=project.coder)
    elif tokens[0] == project.config['string']['as']['directives']['padded']:
        string = line[line.index(tokens[2]):] # Keep multiple spaces
        size = int(tokens[1], 0)
        sequence = process_string(string, project.coder)
        if len(sequence) > size:
            warn(f'Sequence {string} of size {len(sequence)} extends maximal size of {size}. Truncating.')
        else:
            sequence += [0] * (size - len(sequence))
    else:
        return line
    joined = ', '.join(map(str, sequence))
    return f'.byte {joined}'

def preprocess_assembly(input, project, output):
    """ Preprocesses an assembly file. 
    
    Parameters:
    -----------
    input : str
        Path to the input.
    project : pymap.project.Project
        The map project.
    output : str
        Path to the output file.
    """
    with open(input, 'r', encoding='utf-8') as f:
        assembly = f.read()

    # Preprocess assembly linewise
    processed = [preprocess_assembly_line(line, project) for line in split_continued_lines(assembly)] + ['']

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

def preprocess_cpp(input, project, outfile):
    """ Preprocesses a C(++) file.
    
    Parameters:
    -----------
    input : str
        Path to the input.
    project : pymap.project.Project
        The map project.
    outfile : str
        Path to the output file.
    """
    with open(input, 'r', encoding='utf-8') as f:
        source = f.read()
    tail = project.config['string']['tail']
    macro = project.config['string']['c']['macro']

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

            # Parse and concatenate arbitraryly many "..." sequences
            while True:
                # Collect characters until the string is closed
                while offset < len(source):
                    offset += 1
                    if source[offset - 1] == '"':
                        break
                    else:
                        string += source[offset - 1]
                if offset >= len(source):
                    raise Exception(f'Unexpected eof while expecting quotes (start of string)')                
                num_skipped_chars = skip_sementically_irrelevant_characters(source, offset, sementically_irrelevant_characters)
                if offset + num_skipped_chars >= len(source):
                    raise Exception(f'Unexpected eof while expecting ( ')
                offset += num_skipped_chars
                if source[offset] == ')':
                    # No more "..." to parse
                    offset += 1
                    break
                elif source[offset] == '"':
                    # Start next "..." sequence
                    offset += 1
                else:
                    raise Exception(f'Expected ( but got \'{source[offset]}\'')
            output += '{' + ','.join(map(str, project.coder.str_to_hex(string))) + '})'
                
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
