#!/usr/bin/python3

""" This module is responsible for resolving constants and exporting them
as C or assembly macros."""

class Constants:

    def __init__(self, path, macro_configuration):
        """ Initializes a constants instance that can resolve
        and export constants by a .pmp.constants file. The
        manner in which macros are exported and used in source
        files is defined in the macro configuration dict."""
        with open(path, "r") as f:
            self._constants = eval(f.read())
        self.macro_conf = macro_configuration

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
                if pedantic and not value in table["expections"]:
                    raise Exception("Could not resolve value {0} for constant table {1}".format(value, label))
                else: 
                    return hex(value)
            else:
                return values[value - base]
        elif table["type"] == "dict":
            values = table["values"]
            if value not in values:
                # Value not constantizeable
                if pedantic and not value in table["exceptions"]: 
                    raise Exception("Could not resolve value {0} for constant table {1}".format(value, label))
                else: 
                    return hex(value)
            else:
                return values[value]
        else:
            raise Exception("Unkown constant table type {0} for constant table {1}".format(table["type"], label))
    
    def export_macro(self, label, type):
        """ Exports macro definitons for a constant
        table which is placed at the path specified
        by the macro_configuration. A label is passed
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
                macro += format.format(hex(i), const) + "\n"
        elif table["type"] == "dict":
            for i in table["values"]:
                macro += format.format(hex(i), table["values"][i]) + "\n"
        
        conf = self.macro_conf[type]
        path = conf["path"].format(label)
        with open(path, "w+") as f:
            f.write(macro)
            print("Exported macro for constant table {0} to {1}".format(label, path))
    
    def get_include_directive(self, label, type):
        """ Returns the include directive for a particular
        constant table described by its label. The format
        is determined by the type which either must be
        'as' for assembly or 'c' for C language. """
        conf = self.macro_conf[type]
        return conf["directive"].format(label)
        
        
                
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