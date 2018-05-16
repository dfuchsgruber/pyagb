#!/usr/bin/env python3

import sys, getopt, os

def print_usage():
    print("""Usage: bin2s.py [opts] input
Opts:
    -o {output}     :   output path
    -s {symbol}     :   symbol [default=filename
                        without extension]
    """)

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:s:", ["help"])
    except getopt.GetoptError:
        sys.exit(2)
    
    try:
        input = args[0]
    except:
        raise Exception("No input file specified (see --help).")
    
    symbol = None
    output = None

    # Parse opts
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_usage()
            sys.exit(0)
        elif opt == "-s":
            symbol = arg
        elif opt == "-o":
            output = arg
    
    if not output:
        raise Exception("No output file specified (see --help).")

    if not symbol:
        symbol = os.path.splitext(os.path.basename(output))[0]
    
    with open(input, "rb") as f:
        bytes = bytearray(f.read())
    
    # Create assembly
    assembly = ".align 4\n.global {0}\n{0}:\n.byte {1}\n".format(
        symbol, ", ".join(map(hex, bytes))
    )

    with open(output, "w+") as f:
        f.write(assembly)