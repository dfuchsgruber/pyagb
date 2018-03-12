#!/usr/bin/python3

""" This module is responsible for resolving constants and exporting them
as C or assembly macros."""

class Constants:

    def __init__(self, path):
        """ Initializes a constants instance that can resolve
        and export constants by a .pmp.constants file. """
        with open(path, "r") as f:
            self._constants = eval(f.read())

    def constantize(self, value, label, pedantic=False):
        """ Translates a value and returns the corresponding
        constant (string). A label is passed to identify
        the constant table which is used for lookup. If
        the pedantic flag is set the method will raise
        an exception if the value could not be resolved
        and otherwise simply return the values
        hex representation."""
        table = self._constants[label]
        if table["type"] == "enum":
            base = table["base"]
            values = table["values"]
            if value - base not in range(len(values)):
                # Value not constantizeable
                if pedantic: 
                    raise Exception("Could not resolve value {0} for constant table {1}".format(value, label))
                else: 
                    return hex(value)
            else:
                return values[value - base]
        elif table["type"] == "dict":
            values = table["values"]
            if value not in values:
                # Value not constantizeable
                if pedantic: 
                    raise Exception("Could not resolve value {0} for constant table {1}".format(value, label))
                else: 
                    return hex(value)
            else:
                return values[value]
        else:
            raise Exception("Unkown constant table type {0} for constant table {1}".format(table["type"], label))
    
    def export_macro(self, label, type):
        """ Exports macro definitons for a constant
        table which is returned as string. A label is passed
        to identify the constant table. The type parameter
        either is set to 'as' for assembly macros or 'c' for
        C language macros."""
        table = self._constants[label]
        macro = ""
        if type == "as":
            format = ".equ {1}, {0}"
        elif type == "c":
            format = "#define {1} {0}"
        else:
            raise Exception("Unkown macro type {0}".format(type))
        if table["type"] == "enum":
            for i, const in enumerate(table["values"]):
                macro += format.format(i, const) + "\n"
        elif table["type"] == "dict":
            for i in table["values"]:
                macro += format.format(i, table["values"][i]) + "\n"
        return macro
                
    def values(self, label):
        """ Returns all constant symbols that are defined by
        a particular table identified by label. """
        table = self._constants[label]
        if table["type"] == "enum":
            return table["values"]
        elif table["type"] == "dict":
            return table["values"].values()
        else:
            raise Exception("Unkown constant table type {0} for constant table {1}".format(table["type"], label))