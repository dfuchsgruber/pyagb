#!/usr/bin/python3

""" Extension classes for tkinter """
import tkinter
import tkinter.simpledialog
import tkinter.ttk as ttk

class DropDown(tkinter.OptionMenu):
    """ Instance of OptionMenu for better access to attributes"""
    def __init__(self, parent, options: list, initial_value: str=None):
        self.var = tkinter.StringVar(parent)
        self.set(initial_value if initial_value else options[0])
        self.option_menu = tkinter.OptionMenu.__init__(self, parent, self.var, *options)
        self.callback = None

    def add_callback(self, callback: callable):
        """ Adds a callback for when the value of the dropdown changes"""
        def wrapped_callback(*args):
            callback()
        self.wrapped_callback = wrapped_callback
        self.var.trace("w", wrapped_callback)

    def get(self):
        """ Returns the string value of the dropdown"""
        return self.var.get()

    def set(self, value: str):
        """ Sets the string value of the dropdown"""
        self.var.set(value)

class Checkbutton(tkinter.Checkbutton):
    """ Instance of Checkbutton that holds its own int var"""
    def __init__(self, parent, text, onvalue=None, offvalue=None):
        self.var = tkinter.IntVar(parent)
        self.onvalue=onvalue
        self.offvalue=offvalue
        self.checkbutton = tkinter.Checkbutton.__init__(self, parent, variable=self.var, text=text)
        self.trace = True

    def get(self):
        """ Gets the value of the button """
        if self.var.get():
            if self.onvalue == None: return 1
            else: return self.onvalue
        else:
            if self.offvalue == None: return 0
            else: return self.offvalue

    def set(self, value, ignore_trace=False):
        """ Sets the value of the button"""
        if ignore_trace: self.trace = False
        self.var.set(value)
        if ignore_trace: self.trace = True
    
    def add_callback(self, callback: callable):
        """ Adds a callback for when the value of the dropdown changes"""
        def wrapped_callback(*args):
            if self.trace: callback()
        self.var.trace("w", wrapped_callback)

class Entry(tkinter.Entry):
    """ A Wrapper for the tkinter.Entry but allows callbacks at change of the text """
    def __init__(self, parent, callback=None):
        self.sv = tkinter.StringVar(parent)
        super(Entry, self).__init__(parent, textvariable=self.sv)
        self.wrapped_callback = None
        if callback: self.add_callback(callback)

    def get(self):
        """ Gets the text of the entry """
        return self.sv.get()

    def set(self, text, untraced=False):
        """ Sets the text of the entry """
        if untraced: self.disable_trace()
        self.sv.set(text)
        self.delete(0, tkinter.END)
        self.insert(0, text)
        if untraced: self.enable_trace()

    def add_callback(self, callback: callable):
        """ Adds a callback that is called on text change """
        def wrapped_callback(*args):
            callback()
        self.wrapped_callback = wrapped_callback
        self.enable_trace()

    def disable_trace(self):
        """ Disables a trace to the entry """
        if not self.wrapped_callback: return
        self.sv.trace_vdelete("w", self.trace)

    def enable_trace(self):
        """ Reenables the trace to the entry"""
        if not self.wrapped_callback: return
        self.trace = self.sv.trace("w", self.wrapped_callback)      

class Spinbox(tkinter.Spinbox):
    """ Spinbox that supports default and set"""
    def __init__(self, parent, default=0, **kwargs):
        self.sv = tkinter.StringVar(parent, value=default)
        super(Spinbox, self).__init__(parent, textvariable=self.sv, **kwargs)

    def set(self, value):
        self.sv.set(value)


class LabeledSpinbox(Spinbox):
    """ A Wrapper for the tkinterx.Spinbox but shows a label arround (using labelframe) """
    def __init__(self, parent, default=0, text=None, **kwargs):
        self.frame = ttk.Labelframe(parent, text=text)
        super().__init__(self.frame, default=default, **kwargs)
        self.pack()

    def get_handle(self):
        """ Returns the handle (labelframe) for the spinbox """
        return self.frame  

class LabeledEntry(Entry):
    """ A Wrapper for the tkinterx.Entry but shows a label arround (using labelframe)"""
    def __init__(self, parent, text=None, callback=None, **kwargs):
        self.frame = ttk.Labelframe(parent, text=text)
        super(LabeledEntry, self).__init__(self.frame, callback=callback)
        self.pack()

    def get_handle(self):
        """ Returns the handle (labelframe) for the entry """
        return self.frame        


class Combobox(ttk.Combobox):
    """ Wrapper for combobox to match the calling convention for entries """
    def __init__(self, parent, values, callback=None, **kwargs):
        self.sv = tkinter.StringVar(parent)
        super(Combobox, self).__init__(parent, textvariable=self.sv, values=values, **kwargs)
        self.wrapped_callback = None
        if callback: self.add_callback(callback)

    def set(self, text, untraced=False):
        """ Sets the text of the entry """
        if untraced: self.disable_trace()
        self.sv.set(text)
        if untraced: self.enable_trace()

    def add_callback(self, callback: callable):
        """ Adds a callback that is called on text change """
        def wrapped_callback(*args):
            callback()
        self.wrapped_callback = wrapped_callback
        self.enable_trace()

    def disable_trace(self):
        """ Disables a trace to the entry """
        if not self.wrapped_callback: return
        self.sv.trace_vdelete("w", self.trace)

    def enable_trace(self):
        """ Reenables the trace to the entry"""
        if not self.wrapped_callback: return
        self.trace = self.sv.trace("w", self.wrapped_callback)   

class LabeledCombobox(Combobox):
    """ A Wrapper for the ttk.Combobox but shows a label arround (using labelframe)"""
    def __init__(self, parent, values, text=None, callback=None):
        self.frame = ttk.Labelframe(parent, text=text)
        super(LabeledCombobox, self).__init__(self.frame, values, callback=callback)
        self.pack()

    def get_handle(self):
        """ Returns the handle (labelframe) for the entry """
        return self.frame

           
class ComboboxDialog(tkinter.simpledialog.Dialog):
    """ Dialog that """
    def __init__(self, master, values, title, default):
        #Keep values
        self.values = values
        self.default = default
        self.result = None
        super().__init__(master, title=title)

    def body(self, master):
        self.combobox = Combobox(master, self.values)
        self.combobox.set(self.default)
        self.combobox.grid(row=0, column=0)

    def apply(self):
        self.result = self.combobox.get()

def askcombobox(master, title, values, default=""):
    """ Asks for an user input based on a combobox and a set of fixed values"""
    d = ComboboxDialog(master, values, title, default)
    return d.result
