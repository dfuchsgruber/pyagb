""" Class to modell access on behavior """

BEHAVIOR = (0, 8)
HM_USAGE = (8, 13)
FIELD_2 = (13, 17)
FIELD_3 = (17, 23)
FIELD_4 = (23, 26)
FIELD_5 = (26, 28)
FIELD_6 = (28, 31)
FIELD_7 = (31, 32)

def get_field(data, field):
    """ Gets the value of a bitfield inside data (field is a range tuple)"""
    lower, higher = field
    width = higher - lower
    data >>= lower
    data &= 1 << (width - 1)
    return data

def set_field(data, field, value):
    """ Sets the value of a bitfield inside data (field is a range tuple) """
    lower, higher = field
    width = higher - lower
    mask = 1 << (width - 1)
    data &= (~mask) << lower
    data |= (value & mask) << lower
    return data
