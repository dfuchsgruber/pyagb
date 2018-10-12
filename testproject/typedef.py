
# Import the types and structure module
import agb.types
import agb.structure

# Define an array of four u16 values
def size_getter(rom, offset, parent, project):
    return 4 # The array has a constant size of four

my_array_type = agb.types.make_array(agb.types.types['u16'], size_getter)

# Register the new array type
agb.types.types['u16array[4]'] = my_array_type
