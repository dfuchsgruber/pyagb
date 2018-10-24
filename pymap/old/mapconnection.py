OVER, SOUTH, NORTH, WEST, EAST = range(5)


def connection_type_to_adjacence(connection_type):
    """ Returns an adjacence type (OVER, SOUTH, NORTH,
    WEST or EAST) by the connection_type. If no
    adjacence is known for the type it returns OVER"""
    if connection_type in ("SOUTH", 1, "1", "0x1"): return SOUTH
    if connection_type in ("NORTH", 2, "2", "0x2"): return NORTH
    if connection_type in ("WEST", 3, "3", "0x3"): return WEST
    if connection_type in ("EAST", 4, "4", "0x4"): return EAST
    return OVER



class Mapconnection:
    """ Class to modell the map connections  """
    
    def __init__(self):
        self.direction = "NONE"
        self.displacement = 0
        self.bank = 0
        self.mapid = 0
        self.fieldA = 0
        self.fieldB = 0

    def to_dict(self): 
        return self.__dict__


def from_dict(d):
    m = Mapconnection()
    for k, v in d.items(): m.__setattr__(k, v)
    return m
