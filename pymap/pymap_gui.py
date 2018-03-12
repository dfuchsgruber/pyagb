import tkinter
import PIL.Image as PImage
from tkinter import messagebox, filedialog
from . import tkinterx
from . import mapheader, mapfooter, mapevent, tileset, resources, project, tileset_gui, mapconnection, config
import PIL.ImageTk as ImageTk
import os, getopt
from . import tileset_gui
import tkinter.ttk as ttk
import constants

#Global size variables
BLOCK_CANVAS_SIZE = None
MAP_CANVAS_SIZE = None
UNDO_STACK_MAX = 0x10000
#Access to global size variables
WIDTH = 0
HEIGHT = 1


NONE, MAP, LEVEL, EVENTS = range(4)
STDPROJ = config.STDPROJ

class Pymap_gui(tkinter.Frame):
    """ Main gui frame for the pymap module """
    def __init__(self, root, ow_img_pool, proj: project.Project):


        self.map = None
        self.bank, self.mapid = None, None
        self.proj = proj
        self.context = MAP
        self.block_selection_size = 1, 1 #Keeping track of the block selection size
        self.block_selection = [[0]]
        self.block_selection_area = 0, 0, 0, 0
        self._init_gui(root)
        self.undo_stack = [] #Action instance
        self.redo_stack = [] #as above
        self.ow_img_pool = ow_img_pool
        self.show_ow_imgs = True
        self.header_dialog = None
        self.footer_dialog = None
        



    def _init_gui(self, root):
        """ Initializes all members of the gui"""
        tkinter.Frame.__init__(self, root) #Superclass ini
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=10)
        self.root = root
        width, height = root.winfo_screenwidth(), root.winfo_screenheight()

        #Setup the menu
        menu = tkinter.Menu(root)
        root.config(menu=menu)
        file_menu = tkinter.Menu(menu, tearoff=0)
        menu.add_radiobutton(label="Save", command=self.file_save)
        menu.add_radiobutton(label="Header", command=self.edit_header)
        menu.add_radiobutton(label="Footer", command=self.edit_footer)
        menu.add_radiobutton(label="Connections", command=self.edit_connections)
        menu.add_radiobutton(label="Undo", command=self.undo)
        menu.add_radiobutton(label="Redo", command=self.redo)
        overworld_menu = tkinter.Menu(menu, tearoff=0)
        overworld_menu.add_command(label="Images on/off", command=self.show_ow_switch)
        overworld_menu.add_command(label="Show all images", command=self.display_ow_imgs)
        menu.add_cascade(label="Overworld", menu=overworld_menu)

        #Setup the notebook
        self.notebook = ttk.Notebook(self)
        #The map tab
        self.map_frame = tkinter.Frame(self)
        self._setup_block_widget()
        self._setup_map_widget()
        self.level_frame = tkinter.Frame(self)
        self._setup_level_widget()
        self.event_frame = tkinter.Frame(self)
        self._setup_event_widget()
        self.notebook.add(self.map_frame, text="Map")
        self.notebook.add(self.level_frame, text="Level")
        self.notebook.add(self.event_frame, text="Events")
        self.notebook.grid(row=0, column=1, sticky=tkinter.NW)
        self.map_frame.bind("<Visibility>", lambda e: self.context_switch(MAP))
        self.level_frame.bind("<Visibility>", lambda e: self.context_switch(LEVEL))
        self.event_frame.bind("<Visibility>", lambda e: self.context_switch(EVENTS))

        #Setup the map tree
        self.map_tree = resources.ResourcesBrowser(self.root, self.proj, self)
        self.map_tree.grid(column=0, row=0, sticky=tkinter.NW)
        

        self.root.bind_all("<Control-z>", lambda _: self.undo())
        self.root.bind_all("<Control-y>", lambda _: self.redo())
        self.root.bind_all("<Control-f>", lambda _: self.edit_footer())
        self.root.bind_all("<Control-h>", lambda _: self.edit_header())
        self.root.bind_all("<Control-m>", lambda _: self.notebook.select(self.map_frame))
        self.root.bind_all("<Control-e>", lambda _: self.notebook.select(self.event_frame))
        self.root.bind_all("<Control-l>", lambda _: self.notebook.select(self.level_frame))
        
        self.grid(row=0, column=1, sticky=tkinter.NW)


    def context_switch(self, context):
        """ Updates the map of the next visible widget """
        if context == MAP:
            self._refresh_map()
            self.context = MAP
        elif context == LEVEL:
            self._refresh_level_map()
            self.context = LEVEL
        elif context == EVENTS:
            self._refresh_event_map()
            self.context = EVENTS

    def show_ow_switch(self):
        """ Switches the show overworlds mode """
        if not self._can_draw(): return
        self.show_ow_imgs = not self.show_ow_imgs
        self._refresh_event_map()

    def _compute_map_image(self):
        """ Method to initialy compute the map image that lies below """
        if not self._can_draw(): return
        w, h = self.map.footer.width, self.map.footer.height
        self.map_img = PImage.new("RGBA", (w * 16, h * 16), "black")
        for y in range(h):
            for x in range(w):
                block = self.map.footer.blocks[y][x] & 0x3FF
                self.map_img.paste(self.block_imgs[block], (x * 16, y * 16))

    
    def _select_event_by_coordinates(self, x, y):
        """ Selects an map event by coordinates of the canvas """
        if not self._can_draw(): return
        if x in range(self.map.footer.width) and y in range(self.map.footer.height):
            #Check persons
            for i in range(len(self.map.persons)):
                if self.map.persons[i].x == x and self.map.persons[i].y == y: return self._select_event("Person", i)
            for i in range(len(self.map.warps)):
                if self.map.warps[i].x == x and self.map.warps[i].y == y: return self._select_event("Warp", i)
            for i in range(len(self.map.triggers)):
                if self.map.triggers[i].x == x and self.map.triggers[i].y == y: return self._select_event("Trigger", i)
            for i in range(len(self.map.signposts)):
                if self.map.signposts[i].x == x and self.map.signposts[i].y == y: return self._select_event("Sign", i)
        return None


    def _setup_event_widget(self):
        """ Setups the event widets (map and properties)"""
        #First setup the map widget and its event objects
        self.event_map_widget = tkinter.Frame(self.event_frame)
        self.event_map_widget.grid(row=0, column=0, sticky=tkinter.NW)
        self.event_map_widget_canvas = tkinter.Canvas(self.event_map_widget, width=MAP_CANVAS_SIZE[WIDTH], height=MAP_CANVAS_SIZE[HEIGHT], background="lightgray")
        self.event_map_widget_canvas_image = None
        self.event_map_widget_canvas_bar_v = tkinter.Scrollbar(self.event_map_widget, orient=tkinter.VERTICAL)
        self.event_map_widget_canvas_bar_v.grid(row=0, column=1, sticky=tkinter.NSEW)
        self.event_map_widget_canvas_bar_v.config(command=self.event_map_widget_canvas.yview)
        self.event_map_widget_canvas.config(yscrollcommand=self.event_map_widget_canvas_bar_v.set)
        self.event_map_widget_canvas_bar_h = tkinter.Scrollbar(self.event_map_widget, orient=tkinter.HORIZONTAL)
        self.event_map_widget_canvas_bar_h.grid(row=1, column=0, sticky=tkinter.NSEW)
        self.event_map_widget_canvas_bar_h.config(command=self.event_map_widget_canvas.xview)
        self.event_map_widget_canvas.config(xscrollcommand=self.event_map_widget_canvas_bar_h.set)
        self.event_map_widget_canvas.grid(row=0, column=0, sticky=tkinter.NW)
        self.event_map_widget_canvas_selection_rectangles = None
        self.event_map_widget_label_info = tkinter.Label(self.event_map_widget)
        self.event_map_widget_label_info.grid(row=2, column=0, sticky=tkinter.W)

        def update_info(e):
            """ Updates coordinate info """
            if not self._can_draw(): return
            hslider, _ = self.event_map_widget_canvas_bar_h.get()
            vslider, _ = self.event_map_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x, y = int(x / 16), int(y / 16)
            if self._coordinate_in_map(x, y):
                self.event_map_widget_label_info["text"] = "x : " + hex(x) + ", y : " + hex(y)

        
        #variable to indicate weather movements can and should be aggregated (to save space on undo/redo history stack)


        def event_map_canvas_b1_press(e):
            """ Button 1 press on event: select event"""
            if not self._can_draw(): return
            hslider, _ = self.event_map_widget_canvas_bar_h.get()
            vslider, _ = self.event_map_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x, y = int(x / 16), int(y / 16)
            self.selected_event_by_drag = self._select_event_by_coordinates(x, y)
            self.selected_event_by_drag_aggregate = False
        self.event_map_widget_canvas.bind("<Button-1>", event_map_canvas_b1_press)


        def event_map_canvas_b1_motion(e):
            """ Button 1 motion on event: move event """
            if not self._can_draw(): return
            hslider, _ = self.event_map_widget_canvas_bar_h.get()
            vslider, _ = self.event_map_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x, y = int(x / 16), int(y / 16)
            if not self.selected_event_by_drag: return
            if x == self.selected_event_by_drag.x and y == self.selected_event_by_drag.y: return
            if not self._coordinate_in_map(x, y): return
            if self.selected_event_by_drag_aggregate:
                aevent = self.action_peek()
                if not aevent: raise Exception("To aggregate the movement of an event at least one movement should be done before but undo stack is empty!")
                #Aggregate two event movements
                x_middle, x_old = aevent.diffs["x"]
                y_middle, y_old = aevent.diffs["y"]
                aevent.diffs["x"] = x, x_old
                aevent.diffs["y"] = y, y_old
                aevent.do()
            else: 
                self.action(Action_event_change(self, self.event_type, self.event_id, {"x": (x, self.selected_event_by_drag.x), "y" : (y, self.selected_event_by_drag.y)}))
                self.selected_event_by_drag_aggregate = True
            
            #self.selected_event_by_drag.x = x
            #self.selected_event_by_drag.y = y
            #self._refresh_event(self.event_type, self.event_id)
            update_info(e)

        def event_map_canvas_b2_press(e):
            """ Button 1 double click (follow warp if possible)"""
            if not self._can_draw(): return
            event_map_canvas_b1_press(e) #Select the event regulary
            print("double", self.event_type)
            if self.event_type == "Warp":
                self.open_map(self.selected_event_by_drag.target_bank, self.selected_event_by_drag.target_map)
            

        self.event_map_widget_canvas.bind("<B1-Motion>", event_map_canvas_b1_motion)
        self.event_map_widget_canvas.bind("<Motion>", update_info)
        self.event_map_widget_canvas.bind("<Button-2>", event_map_canvas_b2_press)

        #Setup editing widget at border
        self.event_edit_widget = ttk.Labelframe(self.event_frame, text="Properties")
        self.event_edit_widget.grid(row=0, column=1, sticky=tkinter.NW)
        self.event_edit_type_dropdown = tkinterx.DropDown(self.event_edit_widget, ["Person", "Warp", "Trigger", "Sign"], initial_value="None")
        self.event_edit_type_dropdown.add_callback(lambda: self._select_event(self.event_edit_type_dropdown.get(), 0))
        self.event_edit_type_dropdown.grid(row=0, column=0, sticky=tkinter.NW)
        self.event_edit_widgets = []
        self.event_type = "None"
        self.event_id = None

        #Do not setup the ow img widget
        self.event_ow_image_widget = None

    def _select_event(self, type, id):
        """ Selects an event for editing its properties """
        def _build_entries(names, startrow, change_sensitive=False):
            """ Build entries for different attributes in names
            Sets them as attributes: name -> event_edit_widget_{name.lower} for name in names
            Places them at startrow + i for ith element in names
            If change sensitive is chosen the entries trigger an 
            change action whenever a value is entered otherwise only at loss of focus"""
            entries = []
            for name in names:
                widget = tkinterx.LabeledEntry(self.event_edit_widget, text=name)
                self.__setattr__("event_edit_widget_" + name.lower(), widget)
                self.event_edit_widgets.append(widget)
                widget.get_handle().grid(row=startrow, sticky=tkinter.NW)
                startrow += 1
                if change_sensitive: widget.add_callback(lambda: self._event_apply_changes(self.event_type, self.event_id, supress_warnings=True))
                #else: widget.bind("<FocusOut>", (lambda _: self._event_apply_changes(self.event_type, self.event_id, supress_warnings=True)))
                entries.append(widget)
            return entries

        #Transform event type "sign" into more specified version
        if type == "Sign":
            if id in range(len(self.map.signposts)):
                if self.map.signposts[id].structure == mapevent.SIGN_STRUCTURE_SCRIPT: type = "SignScript"
                elif self.map.signposts[id].structure == mapevent.SIGN_STRUCTURE_ITEM: type = "SignItem"
            else: type = "SignScript" # Use any of the sign types since there is no acitve sign
        
        if type == "Person": self.event_edit_type_dropdown.set("Person")
        elif type == "Warp": self.event_edit_type_dropdown.set("Warp")
        elif type == "Trigger": self.event_edit_type_dropdown.set("Trigger")
        elif type in ("Sign", "SignItem", "SignScript"): self.event_edit_type_dropdown.set("Sign")

        #Destroy all widgets on the frame and build those neccessary
        for widget in self.event_edit_widgets:
            try: widget.get_handle().grid_remove() #Try to remove as tkinterx.LabeledEntry
            except: widget.grid_remove()
            try: self.entry_widget_behaviour_dropdown.grid_remove()
            except: pass
        self.event_edit_widgets = []
        self.event_type = type
        if self.event_type == "Person": #Build widgets for person
            if len(self.map.persons) <= id: event_present = False
            else:
                event_present = True
                self.event_edit_widget_id_spinbox = tkinterx.Spinbox(self.event_edit_widget, from_=0, to=len(self.map.persons) - 1, default=id, command=lambda: self._select_event(self.event_type, int(self.event_edit_widget_id_spinbox.get())))
                entries = _build_entries(["Id"], 4)
                entries += _build_entries(["Picture"], 5, change_sensitive=True)
                entries += _build_entries(["Field2", "Field3", "Level"], 6)
                #Behaviour is a combo box and thus different
                cbox_behaviour = tkinterx.LabeledCombobox(self.event_edit_widget, text="Behaviour", values=constants.values(constants.behaviours))
                self.__setattr__("event_edit_widget_" + "Behaviour".lower(), cbox_behaviour)
                cbox_behaviour.get_handle().grid(row=9, sticky=tkinter.NW)
                #cbox_behaviour.bind("<FocusOut>", (lambda _: self._event_apply_changes(self.event_type, self.event_id, supress_warnings=True)))
                entries.append(cbox_behaviour)
                self.event_edit_widgets.append(cbox_behaviour)
                entries += _build_entries(["Behaviour range", "FieldB", "Is Trainer", "Padding", "FieldD", "Alert range", "Script", "Flag", "Field16", "Field17"], 10)
            
               
        elif self.event_type == "Warp": #Build widgets for warp
            if len(self.map.warps) <= id: event_present = False
            else:
                event_present = True
                self.event_edit_widget_id_spinbox = tkinterx.Spinbox(self.event_edit_widget, from_=0, to=len(self.map.warps) - 1, default=id, command=lambda: self._select_event(self.event_type, int(self.event_edit_widget_id_spinbox.get())))
                entries = _build_entries(["Level", "Warp", "Bank", "Map"], 4)
        elif self.event_type == "Trigger": #Build widgets for trigger
            if len(self.map.triggers) <= id: event_present = event_present = False
            else:
                event_present = True
                self.event_edit_widget_id_spinbox = tkinterx.Spinbox(self.event_edit_widget, from_=0, to=len(self.map.triggers) - 1, default=id, command=lambda: self._select_event(self.event_type, int(self.event_edit_widget_id_spinbox.get())))
                entries = _build_entries(["Level", "Field5", "Variable", "Value", "FieldA", "FieldB", "Script"], 4)
        elif self.event_type == "SignScript": #Build widgets for sign (script structure)
            if len(self.map.signposts) <= id: event_present = False
            else:
                event_present = True
                self.event_edit_widget_id_spinbox = tkinterx.Spinbox(self.event_edit_widget, from_=0, to=len(self.map.signposts) - 1, default=id, command=lambda: self._select_event("Sign", int(self.event_edit_widget_id_spinbox.get())))
                entries = _build_entries(["Level", "Type", "Field6", "Field7", "Script"], 4)
        elif self.event_type == "SignItem":
            if len(self.map.signposts) <= id: event_present = False
            else:
                event_present = True
                self.event_edit_widget_id_spinbox = tkinterx.Spinbox(self.event_edit_widget, from_=0, to=len(self.map.signposts) - 1, default=id, command=lambda: self._select_event("Sign", int(self.event_edit_widget_id_spinbox.get())))
                entries = _build_entries(["Level", "Type", "Field6", "Field7", "Item", "Hidden Id", "Flags, Chunk, Count"], 4)
        elif self.event_type == "None": return
        else: 
            raise Exception("Unkown event type selected: " + self.event_type)
        if event_present:
            #Push all common widgets to the ui
            self.event_edit_widget_id_spinbox.grid(row=1, column=0, columnspan=1, sticky=tkinter.NW)
            self.event_edit_widget_id_spinbox.config(state="readonly")
            self.event_edit_widgets.append(self.event_edit_widget_id_spinbox)
            #Coordinate entry is common for all event types
            entries = _build_entries(["X", "Y"], 2, change_sensitive=True) + entries
            #Apply button
            apply_button = tkinter.Button(self.event_edit_widget, text="Apply", command=lambda: self._event_apply_changes(self.event_type, id))
            apply_button.grid(row=len(self.event_edit_widgets) + 2, sticky=tkinter.NW)
            self.event_edit_widgets.append(apply_button)
            #Remove button
            remove_button = tkinter.Button(self.event_edit_widget, text="Remove (this)", command=lambda: self._remove_event(self.event_type, id))
            remove_button.grid(row=len(self.event_edit_widgets) + 2, sticky=tkinter.NW)
            self.event_edit_widgets.append(remove_button)
        else:
            self.event_id = None

        #New button (present even when there are no more events)
        new_button = tkinter.Button(self.event_edit_widget, text="New (append)", command=lambda: self._add_new_event(self.event_type))
        new_button.grid(row=len(self.event_edit_widgets) + 2, sticky=tkinter.NW)
        self.event_edit_widgets.append(new_button)
        if event_present:
            self.event_id = id
            if type == "Person": event = self.map.persons[id]
            elif type == "Warp": event = self.map.warps[id]
            elif type == "Trigger": event = self.map.triggers[id]
            elif type in ("SignScript", "SignItem", "Sign"): event = self.map.signposts[id]
            self._event_edit_sync_entries_with_active_event(type, id)
            self._refresh_event_edit_active_rectangle()
            return event
        else: return None

    def _remove_event(self, stype, id):
        """ Removes an event """
        if not self._can_draw(): return
        self.action(Action_event_remove(self, stype, id))
        """
        if stype == "Person":
            self.map.persons.remove(self.map.persons[id])
            if id >= len(self.map.persons): id = len(self.map.persons) - 1
        elif stype == "Warp":
            self.map.warps.remove(self.map.warps[id])
            if id >= len(self.map.warps): id = len(self.map.warps) - 1
        elif stype == "Trigger":
            self.map.triggers.remove(self.map.triggers[id])
            if id >= len(self.map.triggers): id = len(self.map.triggers) - 1
        elif stype in ("Sign", "SignScript", "SignItem"):
            self.map.signposts.remove(self.map.signposts[id])
            if id >= len(self.map.signposts): id = len(self.map.signposts) - 1
        if id < 0:
            self._select_event(stype, 0)
        else:
            self._select_event(stype, id)
        self._refresh_event_map()
        """


    def _add_new_event(self, stype):
        """ Adds a new event """
        if not self._can_draw(): return
        self.action(Action_event_add(self, stype))
        """if stype == "Person":
            if len(self.map.persons) >= 256: messagebox.showwarning("Event overflow", "Using more than 256 instances of the same event might cause trouble when accessing events")
            self.map.persons.append(mapevent.Map_event_person())
            self._refresh_event_map()
            self._select_event(stype, len(self.map.persons) - 1)
        elif stype == "Warp":
            if len(self.map.warps) >= 256: messagebox.showwarning("Event overflow", "Using more than 256 instances of the same event might cause trouble when accessing events")
            self.map.warps.append(mapevent.Map_event_warp())
            self._refresh_event_map()
            self._select_event(stype, len(self.map.warps) - 1)
        elif stype == "Trigger":
            if len(self.map.triggers) >= 256: messagebox.showwarning("Event overflow", "Using more than 256 instances of the same event might cause trouble when accessing events")
            self.map.triggers.append(mapevent.Map_event_trigger())
            self._refresh_event_map()
            self._select_event(stype, len(self.map.triggers) - 1)
        elif stype in ("Sign", "SignScript", "SignItem"):
            if len(self.map.signposts) >= 256: messagebox.showwarning("Event overflow", "Using more than 256 instances of the same event might cause trouble when accessing events")
            self.map.signposts.append(mapevent.Map_event_sign())
            self._refresh_event_map()
            self._select_event(stype, len(self.map.signposts) - 1)
        """

    def _event_edit_sync_entries_with_active_event(self,stype, id):
        """ Syncs all entries in the properties widget with one event """
        def _load_entires(event, assocs):
            """ Updates entires with values of attributes of an event.
            assocs is a dict entry_name(as in _build_entries) -> attribute_name"""
            for entry_name in assocs:
                widget = self.__getattribute__("event_edit_widget_" + entry_name.lower())
                attribute = event.__getattribute__(assocs[entry_name])
                try: widget.set(hex(attribute), untraced=True)
                except: widget.set(str(attribute), untraced=True)
        if stype == "Person":
            event = self.map.persons[id]
            _load_entires(self.map.persons[id], {"Id" : "target_index", "Picture" : "picture", "Field2": "field_2", "Field3" : "field_3", "X" : "x", "Y" : "y", "Level" : "level", "Behaviour" : "behaviour", "Is Trainer" : "is_trainer",
            "Padding" : "is_trainer_padding", "FieldD" : "field_d", "FieldB" : "field_b", "Alert range" : "alert_radius", "Script" : "script", "Flag" : "flag", "Field16" : "field_16", "Field17" : "field_17", "Behaviour range" : "behaviour_range"})
        elif stype == "Warp":
            event = self.map.warps[id]
            _load_entires(self.map.warps[id], {"X" : "x", "Y" : "y", "Level" : "level", "Warp" : "target_warp", "Bank" : "target_bank", "Map" : "target_map"})
        elif stype == "Trigger":
            event = self.map.triggers[id]
            _load_entires(self.map.triggers[id], {"X" : "x", "Y" : "y", "Level" : "level", "Field5" : "field_5", "Variable" : "variable", "Value" : "value", "FieldA" : "field_a", "FieldB" : "field_b", "Script" : "script"})
        elif stype == "SignScript":
            event = self.map.signposts[id]
            _load_entires(self.map.signposts[id], {"X" : "x", "Y" : "y", "Level" : "level", "Type" : "sign_type", "Field6" : "field_6", "Field7" : "field_7", "Script" : "script"})
        elif stype == "SignItem":
            event = self.map.signposts[id]
            _load_entires(self.map.signposts[id], {"X" : "x", "Y" : "y", "Level" : "level", "Type" : "sign_type", "Field6" : "field_6", "Field7" : "field_7", "Item" : "item_id", "Hidden Id" : "hidden", "Flags, Chunk, Count" : "count"})
        else: raise Exception("Unkown event type " + stype)

    def _event_apply_changes(self, stype, id, supress_warnings=False):
        """ Applies changes to the currently selected event by its id, specified type """
        if not self._can_draw(): return
        if stype == "Person":
            assocs = {"Id" : "target_index", "Picture" : "picture", "Field2": "field_2", "Field3" : "field_3", "X" : "x", "Y" : "y", "Level" : "level", "Behaviour" : "behaviour", "Is Trainer" : "is_trainer",
            "Padding" : "is_trainer_padding", "FieldD" : "field_d", "FieldB" : "field_b", "Alert range" : "alert_radius", "Script" : "script", "Flag" : "flag", "Field16" : "field_16", "Field17" : "field_17", "Behaviour range" : "behaviour_range"}
            event = self.map.persons[id]
            #self._refresh_event_person_picture(id)
        elif stype == "Warp":
            assocs = {"X" : "x", "Y" : "y", "Level" : "level", "Warp" : "target_warp", "Bank" : "target_bank", "Map" : "target_map"}
            event = self.map.warps[id]
        elif stype == "Trigger":
            assocs = {"X" : "x", "Y" : "y", "Level" : "level", "Field5" : "field_5", "Variable" : "variable", "Value" : "value", "FieldA" : "field_a", "FieldB" : "field_b", "Script" : "script"}
            event = self.map.triggers[id]
        elif stype == "SignScript":
            assocs = {"X" : "x", "Y" : "y", "Level" : "level", "Type" : "sign_type", "Field6" : "field_6", "Field7" : "field_7", "Script" : "script"}
            event = self.map.signposts[id]
        elif stype == "SignItem":
            assocs =  {"X" : "x", "Y" : "y", "Level" : "level", "Type" : "sign_type", "Field6" : "field_6", "Field7" : "field_7", "Item" : "item_id", "Hidden Id" : "hidden", "Flags, Chunk, Count" : "count"}
            event = self.map.signposts[id]
        else: raise Exception("Unknown specified event type " + stype)
        diffs = {}
        for entry_name in assocs:
            widget = self.__getattribute__("event_edit_widget_" + entry_name.lower())
            attribute = event.__getattribute__(assocs[entry_name])
            new_value = widget.get()
            #Special types required for attributes X, Y, Picture
            try: new_value = int(widget.get(), 0)
            except Exception as e:
                if entry_name in ("X", "Y", "Picture", "Type"): 
                    if not supress_warnings: messagebox.showerror("Invalid property", "Required integer value for " + entry_name + " Exception: " + str(e))
                    return
            #event.__setattr__(assocs[entry_name], new_value)
            if attribute != new_value: diffs[assocs[entry_name]] = new_value, attribute
        if len(diffs):
            #print(diffs)
            self.action(Action_event_change(self, stype, id, diffs))


    def _refresh_event(self, stype, id):
        """ Refreshes the picture of a particular event on the map """
        if not self._can_draw(): return
        image = False
        if stype == "Person":
            event = self.map.persons[id]
            if self.show_ow_imgs:
                img, width, height = self.event_map_widget_canvas_person_objects[id]
                image = True
            else:
                text, rect = self.event_map_widget_canvas_person_objects[id]
        elif stype == "Warp":
            event = self.map.warps[id]
            text, rect = self.event_map_widget_canvas_warp_objects[id]
        elif stype == "Trigger":
            event = self.map.triggers[id]
            text, rect = self.event_map_widget_canvas_trigger_objects[id]
        elif stype in ("Sign", "SignItem", "SignScript"):
            event = self.map.signposts[id]
            text, rect = self.event_map_widget_canvas_sign_objects[id]
        else: raise Exception("Unkown specified event type " + stype)
        x, y = event.x, event.y
        if image:
            self.event_map_widget_canvas.coords(img, x * 16 - (width >> 1) + 8, y * 16 - (height) + 16)
        else:
            self.event_map_widget_canvas.coords(text, x * 16 + 8, y * 16 + 8)
            self.event_map_widget_canvas.coords(rect, 16 * x, 16 * y, 16 * (x + 1), 16 * (y + 1))
        self._refresh_event_edit_active_rectangle()
        

    def _setup_level_widget(self):
        """ Setups the level widgets (map and selection) """
        #First setup the map widget
        self.level_widget = tkinter.Frame(self.level_frame)
        self.level_widget.grid(row=0, column=0, sticky=tkinter.NW)
        self.level_widget_canvas = tkinter.Canvas(self.level_widget, width = MAP_CANVAS_SIZE[WIDTH], height=MAP_CANVAS_SIZE[HEIGHT], background="lightgray")
        self.level_widget_canvas_image = None
        self.level_widget_canvas_bar_v = tkinter.Scrollbar(self.level_widget, orient=tkinter.VERTICAL)
        self.level_widget_canvas_bar_v.grid(row=1, column=1, sticky=tkinter.NSEW)
        self.level_widget_canvas_bar_v.config(command=self.level_widget_canvas.yview)
        self.level_widget_canvas.config(yscrollcommand=self.level_widget_canvas_bar_v.set)
        self.level_widget_canvas_bar_h = tkinter.Scrollbar(self.level_widget, orient=tkinter.HORIZONTAL)
        self.level_widget_canvas_bar_h.grid(row=2, column=0, sticky=tkinter.NSEW)
        self.level_widget_canvas_bar_h.config(command=self.level_widget_canvas.xview)
        self.level_widget_canvas.config(xscrollcommand=self.level_widget_canvas_bar_h.set)
        self.level_widget_canvas.grid(row=1, column=0)
        self.level_widget_label_info = tkinter.Label(self.level_widget)
        self.level_widget_label_info.grid(row=3, column=0, sticky=tkinter.W)

        #Setup the border widget for selection
        self.level_selection_widget = ttk.Labelframe(self.level_frame, text="Levels")
        self.level_selection_widget.grid(row=0, column=1, sticky=tkinter.NW)
        self.level_selection_canvas = tkinter.Canvas(self.level_selection_widget, width=16, height=MAP_CANVAS_SIZE[HEIGHT], background="grey", scrollregion=(0, 0, 16, 0x40 * 16))
        self.level_selection_canvas.pack(side=tkinter.LEFT)
        self.level_selection_canvas_bar_v = tkinter.Scrollbar(self.level_selection_widget, orient=tkinter.VERTICAL)
        self.level_selection_canvas_bar_v.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.level_selection_canvas_bar_v.config(command=self.level_selection_canvas.yview)
        self.level_selection_canvas.config(yscrollcommand=self.level_selection_canvas_bar_v.set)
        self.level_selection_canvas_items = []
        for y in range(0x40):
            rect = self.level_selection_canvas.create_rectangle(0, 16 * y, 16, 16 * (y+1), fill=self.color_by_level(y))
            text = self.level_selection_canvas.create_text(8, 16 * y + 8, text=_strcap(hex(y)[2:]), anchor=tkinter.CENTER, justify=tkinter.CENTER, width=16)
            self.level_selection_canvas_items.append((text, rect))
        self.level_selection_canvas.itemconfig(self.level_selection_canvas_items[0][0], fill="white")
        self.level_selection = 0

        self.level_last_applied_coords = None

        def level_widget_canvas_motion(e):
            """ Motion on map level canavs -> level update """
            if not self._can_draw(): return
            hslider, _ = self.level_widget_canvas_bar_h.get()
            vslider, _ = self.level_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x0, y0 = int(x / 16), int(y / 16)
            if not self._coordinate_in_map(x0, y0): return
            level = self.map.footer.blocks[y0][x0] >> 10
            _update_level(level, x0, y0)

        def _update_level(level, x, y):
            """ Updates info on current level """
            level_info = ["accessible", "solid", "???(02)", "???(03)"]
            if level < 4:
                info = ["level connections", "always solid", "???(02)", "???(03)"][level]
            else:
                info = "height " + str(level >> 2) + ", " + level_info[level & 3]
                if level == 4: info += " (surf)"
            if x != None and y != None:
                self.level_widget_label_info["text"] = "x : " + hex(x) + ", y : " + hex(y) + ", " + info
            else:
                self.level_widget_label_info["text"] = info
                

        def level_widget_canvas_b1_press(e):
            """ Left mouse click on canvas -> Change level"""
            if not self._can_draw(): return
            hslider, _ = self.level_widget_canvas_bar_h.get()
            vslider, _ = self.level_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x0, y0 = int(x / 16), int(y / 16)
            if not self._coordinate_in_map(x0, y0): return
            if self.level_last_applied_coords != None:
                xl, yl = self.level_last_applied_coords
                if x0 == xl and y0 == yl: return
            self.level_last_applied_coords = x0, y0
            self.action(Action_level_change(self, x0, y0, self.level_selection))

        def level_widget_canvas_b3_press(e):
            """ Right mouse click on canvas -> pickup level """
            if not self._can_draw(): return
            hslider, _ = self.level_widget_canvas_bar_h.get()
            vslider, _ = self.level_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x0, y0 = int(x / 16), int(y / 16)
            if not self._coordinate_in_map(x0, y0): return
            level = self.map.footer.blocks[y0][x0] >> 10
            _update_level(level, x0, y0)
            if level == self.level_selection: return
            self.level_selection_canvas.itemconfig(self.level_selection_canvas_items[self.level_selection][0], fill="black")
            self.level_selection = level
            self.level_selection_canvas.itemconfig(self.level_selection_canvas_items[level][0], fill="white")
            #Adjust the scrollbar
            slider_top = level / 0x40
            displayed = MAP_CANVAS_SIZE[HEIGHT] / (0x40 * 16)
            not_displayed = 1 - displayed
            if slider_top > not_displayed: slider_top = not_displayed
            slider_bottom = slider_top + displayed
            self.level_selection_canvas_bar_v.set(slider_top, slider_bottom)
            self.level_selection_canvas.yview("moveto", slider_top)

        def level_widget_canvas_b2_press(e):
            """ Middle mouse click on canvas -> fill level"""
            if not self._can_draw(): return
            hslider, _ = self.level_widget_canvas_bar_h.get()
            vslider, _ = self.level_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x0, y0 = int(x / 16), int(y / 16)
            if not self._coordinate_in_map(x0, y0): return
            level = self.level_selection
            self.action(Action_level_fill(self, x0, y0, level))

        def level_selection_canvas_b_press(e):
            """ Button press on level selection canvas """
            if not self._can_draw(): return
            vslider, _ = self.level_selection_canvas_bar_v.get()
            y = e.y + (vslider * 16 * 0x40)
            level = int(y / 16)
            if level == self.level_selection: return
            self.level_selection_canvas.itemconfig(self.level_selection_canvas_items[self.level_selection][0], fill="black")
            self.level_selection = level
            self.level_selection_canvas.itemconfig(self.level_selection_canvas_items[level][0], fill="white")

        def level_selection_canvas_motion(e):
            """ Motion on selection canvas -> update level info"""
            if not self._can_draw(): return
            vslider, _ = self.level_selection_canvas_bar_v.get()
            y = e.y + (vslider * 16 * 0x40)
            level = int(y / 16)
            if level >= 0x40: return
            _update_level(level, None, None)

        self.level_widget_canvas.bind("<ButtonPress-1>", level_widget_canvas_b1_press)
        self.level_widget_canvas.bind("<B1-Motion>", level_widget_canvas_b1_press)
        self.level_widget_canvas.bind("<ButtonPress-3>", level_widget_canvas_b3_press)
        self.level_widget_canvas.bind("<ButtonPress-2>", level_widget_canvas_b2_press)
        self.level_widget_canvas.bind("<B3-Motion>", level_widget_canvas_b3_press)
        self.level_widget_canvas.bind("<Motion>", level_widget_canvas_motion)
        self.level_selection_canvas.bind("<ButtonPress-1>", level_selection_canvas_b_press)
        self.level_selection_canvas.bind("<ButtonPress-3>", level_selection_canvas_b_press)
        self.level_selection_canvas.bind("<Motion>", level_selection_canvas_motion)


    def _setup_map_widget(self):
        """ Creates the map widget in the center of the window in the context of blocks"""
        self.map_widget = tkinter.Frame(self.map_frame)
        self.map_widget.grid(row=0, column=0, sticky=tkinter.NW)
        self.map_widget_canvas = tkinter.Canvas(self.map_widget, width = MAP_CANVAS_SIZE[WIDTH], height=MAP_CANVAS_SIZE[HEIGHT], background="lightgray")
        self.map_widget_canvas_image = None
        self.map_widget_canvas_bar_v = tkinter.Scrollbar(self.map_widget, orient=tkinter.VERTICAL)
        self.map_widget_canvas_bar_v.grid(row=1, column=1, sticky=tkinter.NSEW)
        self.map_widget_canvas_bar_v.config(command=self.map_widget_canvas.yview)
        self.map_widget_canvas.config(yscrollcommand=self.map_widget_canvas_bar_v.set)
        self.map_widget_canvas_bar_h = tkinter.Scrollbar(self.map_widget, orient=tkinter.HORIZONTAL)
        self.map_widget_canvas_bar_h.grid(row=2, column=0, sticky=tkinter.NSEW)
        self.map_widget_canvas_bar_h.config(command=self.map_widget_canvas.xview)
        self.map_widget_canvas.config(xscrollcommand=self.map_widget_canvas_bar_h.set)
        self.map_widget_canvas.grid(row=1, column=0)
        self.map_widget_label_info = tkinter.Label(self.map_widget)
        self.map_widget_label_info.grid(row=3, column=0, sticky=tkinter.W)
        def _update_info(event):
            """ Updates infos concerning coordinates and blocks """
            if not self._can_draw(): return
            x, y = int(event.x / 16), int(event.y / 16)
            if self._coordinate_in_map(x, y):
                block = self.map.footer.blocks[y][x] & 0x3FF
                self.map_widget_label_info["text"] = "x : " + hex(x) + ", y : " + hex(y) + ", Block : " + hex(block)

        def _map_canvas_b3_press(e):
            """ Right mouse click on map canvas -> start picking up tiles """
            if not self._can_draw(): return
            hslider, _ = self.map_widget_canvas_bar_h.get()
            vslider, _ = self.map_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x0, y0 = int(x / 16), int(y / 16)
            if not self._coordinate_in_map(x0, y0): return
            self.update_block_selection_by_map_area(x0, y0, x0 + 1, y0 + 1)
            #Update the scrollbars in the canvas to show block
            block = self.block_selection[0][0] & 0x3FF
            y0 = int(block / 8)
            slider_top = y0 / 128
            displayed = MAP_CANVAS_SIZE[HEIGHT] / 2048
            not_displayed = 1 - displayed
            if slider_top > not_displayed: slider_top = not_displayed
            slider_bottom = slider_top + displayed
            self.block_canvas_bar_v.set(slider_top, slider_bottom)
            self.block_canvas.yview("moveto", slider_top)

        def _map_canvas_b3_motion(e):
            """ Right mouse motion on map canvas -> picking up tiles """
            if not self._can_draw(): return
            _update_info(e) #We want to keep coordinates on track as well
            x0, y0, old_x1, old_y1 = self.map_selection_area
            hslider, _ = self.map_widget_canvas_bar_h.get()
            vslider, _ = self.map_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x1, y1 = int(x / 16), int(y / 16)
            if not self._coordinate_in_map(x1, y1): return
            if x1 > self.map.footer.width: x1 = self.map.footer.width
            if y1 > self.map.footer.height: y1 = self.map.footer.height
            if x1 <= x0: x1 = x0 + 1
            if y1 <= y0: y1 = y0 + 1
            if x1 == old_x1 and y1 == old_y1: return #No changes
            self.update_block_selection_by_map_area(x0, y0, x1, y1)

        def _map_canvas_b1_press(e):
            """ Left mouse click on map canvas -> Paste selection"""
            if not self._can_draw(): return
            hslider, _ = self.map_widget_canvas_bar_h.get()
            vslider, _ = self.map_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x0, y0 = int(x / 16), int(y / 16)
            if not self._coordinate_in_map(x0, y0): return
            self.map_canvas_last_coords = x0, y0
            self.action(Action_map_change(self, x0, y0, self.block_selection))

        def _map_canvas_b1_motion(e):
            """ Left mouse motion on  map canvas -> Paste selection if not already pasted there """
            if not self._can_draw(): return
            _update_info(e) #We want to keep coordinates on track as well
            if len(self.block_selection) > 1 or max([len(line) for line in self.block_selection]) > 1: return #Making selections bigger than 1x1 draggable makes mapping uncomfortable
            hslider, _ = self.map_widget_canvas_bar_h.get()
            vslider, _ = self.map_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x0, y0 = int(x / 16), int(y / 16)
            if not self._coordinate_in_map(x0, y0): return
            x0l, y0l = self.map_canvas_last_coords
            if x0 != x0l or y0 != y0l:
                self.map_canvas_last_coords = x0, y0
                self.action(Action_map_change(self, x0, y0, self.block_selection))
        
        def _map_canvas_b2_press(e):
            """ Middle mouse click on  map canvas -> Fill if the selection is 1x1 """
            if not self._can_draw(): return
            if len(self.block_selection) > 1 or max([len(line) for line in self.block_selection]) > 1: return
            hslider, _ = self.map_widget_canvas_bar_h.get()
            vslider, _ = self.map_widget_canvas_bar_v.get()
            x, y = e.x + (hslider * self.map.footer.width * 16), e.y + (vslider * self.map.footer.height * 16)
            x0, y0 = int(x / 16), int(y / 16)
            if not self._coordinate_in_map(x0, y0): return
            self.action(Action_map_fill(self, x0, y0, self.block_selection[0][0] & 0x3FF))
            

        self.map_widget_canvas.bind("<Motion>", _update_info)
        self.map_widget_canvas.bind("<ButtonPress-3>", _map_canvas_b3_press)
        self.map_widget_canvas.bind("<B3-Motion>", _map_canvas_b3_motion)
        self.map_widget_canvas.bind("<ButtonPress-1>", _map_canvas_b1_press)
        self.map_widget_canvas.bind("<B1-Motion>", _map_canvas_b1_motion)
        self.map_widget_canvas.bind("<ButtonPress-2>", _map_canvas_b2_press)


    def _setup_block_widget(self):
        """ Creates the block widget at the right part of the window """
        self.block_widget = tkinter.Frame(self.map_frame)
        self.block_widget.grid(row=0, column=1, sticky=tkinter.NW)

        self.block_widget_block_pane = ttk.Labelframe(self.block_widget, text="Blocks")
        self.block_widget_block_pane.grid(row=0, column=0, sticky=tkinter.NW)
        self.block_widget_selection_pane = ttk.Labelframe(self.block_widget, text="Selection")
        self.block_widget_selection_pane.grid(row=1, column=0, sticky=tkinter.NW)
        self.block_widget_border_pane = ttk.Labelframe(self.block_widget, text="Border")
        self.block_widget_border_pane.grid(row=2, column=0, sticky=tkinter.NW)

        self.border_canvas = tkinter.Canvas(self.block_widget_border_pane, width=16 * 1, height=16* 1, background="black")
        self.border_canvas.grid(row=0, column=0, sticky=tkinter.NW)

        self.selection_frame = tkinter.Frame(self.block_widget_selection_pane)
        self.selection_frame.grid(row=0, column=0, sticky=tkinter.NW)
        self.selection_canvas = tkinter.Canvas(self.selection_frame, width=16, height=16, background="black")
        self.selection_raw_pil_image = PImage.new("RGBA", (16, 16), "black")
        self.selection_raw_image = ImageTk.PhotoImage(self.selection_raw_pil_image)
        self.selection_canvas.pack(side=tkinter.LEFT)
        self.selection_canvas_image = self.selection_canvas.create_image(1, 1, image=self.selection_raw_image, anchor=tkinter.NW)
        
        self.block_canvas_frame = tkinter.Frame(self.block_widget_block_pane)
        self.block_canvas_frame.grid(sticky=tkinter.NW)
        self.block_canvas = tkinter.Canvas(self.block_canvas_frame, width=128, height=BLOCK_CANVAS_SIZE[HEIGHT], background="black", scrollregion=(0, 0, 128, 2048))
        self.block_canvas.pack(side=tkinter.LEFT)
        self.block_canvas_bar_v = tkinter.Scrollbar(self.block_canvas_frame, orient=tkinter.VERTICAL)
        self.block_canvas_bar_v.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.block_canvas_bar_v.config(command=self.block_canvas.yview)
        self.block_canvas.config(yscrollcommand=self.block_canvas_bar_v.set)
        self.block_raw_pil_image = PImage.new("RGBA", (128, 1024), "black")
        self.block_raw_image = ImageTk.PhotoImage(self.block_raw_pil_image)
        self.block_canvas_image = self.block_canvas.create_image(1, 1, image=self.block_raw_image, anchor=tkinter.NW)
        self.block_canvas_selection_rectangle = self.block_canvas.create_rectangle(0, 0, 16, 16, outline="red", fill="")

        #Event binding of block canvas (selection)
        

        def block_canvas_b_pressed(e):
            slider_top, slider_bottom = self.block_canvas_bar_v.get()
            x, y = e.x, e.y + (slider_top * 2048)
            x0, y0 = int(x / 16), int(y / 16)
            self.update_block_selection_by_block_area(x0, y0, x0 + 1, y0 + 1)
        
        def block_canvas_b_motion(e):
            x0, y0, _, _ = self.block_selection_area
            slider_top, slider_bottom = self.block_canvas_bar_v.get()
            x, y = e.x, e.y + (slider_top * 2048)
            x1, y1 = int(x / 16) + 1, int(y / 16) + 1
            if x1 > 8: x1 = 8
            if y1 > 256: y1 = 256
            if x1 <= x0: x1 = x0 + 1
            if y1 <= y0: y1 = y0 + 1
            self.update_block_selection_by_block_area(x0, y0, x1, y1)


        self.block_canvas.bind("<ButtonPress-1>", block_canvas_b_pressed)
        self.block_canvas.bind("<ButtonPress-3>", block_canvas_b_pressed)
        self.block_canvas.bind("<B3-Motion>", block_canvas_b_motion)

    def update_block_selection_by_block_area(self, x0, y0, x1, y1):
        """ Updates the block selection by the block selection area """
        if not self._can_draw(): return
        #Check if the rectangle is valid (bottom right block is defined)
        if (y1 - 1) * 8 + x1 - 1 >= len(self.block_imgs): return
        self.block_selection_area = x0, y0, x1, y1
        self.block_selection = [[8 * y + x for x in range(x0, x1)] for y in range(y0, y1)]
        self.block_canvas.delete(self.block_canvas_selection_rectangle)
        self.block_canvas_selection_rectangle = self.block_canvas.create_rectangle(16 * x0, 16 * y0, 16 * x1, 16 * y1, fill="", outline="red")
        self._refresh_block_selection()

    def update_block_selection_by_map_area(self, x0, y0, x1, y1):
        """ Updates the block selection by the map selection area """
        if not self._can_draw(): return
        #Check if the rectangle is valid (bottom right block is defined)
        if (y1 - 1) >= self.map.footer.height or (x1 - 1) >= self.map.footer.width: return
        self.map_selection_area = x0, y0, x1, y1
        #If a 1x1 area is selected we can treat is as block selection
        if y1 - 1 == y0 and x1 - 1 == x0:
            block = self.map.footer.blocks[y0][x0] & 0x3FF
            bx, by = block % 8, int(block / 8)
            self.update_block_selection_by_block_area(bx, by, bx + 1, by + 1)
        else:
            self.block_selection = [[self.map.footer.blocks[y][x] & 0x3FF for x in range(x0, x1)] for y in range(y0, y1)]
            self._refresh_block_selection()
        
            


    def _prepeare_undo_memory(self):
        """ Method to prevent the undo stack from being limetless
        By defining a maximum height and calling this before every push
        by deleting the bottom element the stack never exceeds its height """
        while len(self.undo_stack) >= UNDO_STACK_MAX: self.undo_stack.remove(0)

    def action(self, action):
        """ Executes an undoable action and pushes it to the undo stack
        It also clears the redo stack. """
        print("action", str(action.__class__))
        self._prepeare_undo_memory()
        self.undo_stack.append(action)
        action.do()
        self.redo_stack = []

    def action_peek(self):
        """ Peeks from the undo/redo stack history and returns None if emtpy """
        if len(self.undo_stack): return self.undo_stack[-1]
        else: return None

    def _can_draw(self):
        """ Checks if anything can drawn (juding from state of tss and tsp)"""
        if not self.map: return False
        if not self.map.footer: return False
        if not self.map.footer.tsp or not self.map.footer.tss: return False
        if self.map.footer.tsp.image.empty or self.map.footer.tss.image.empty: return False
        return True
    def _coordinate_in_map(self, x, y):
        """ Checks if certain coordinates are in range of a map """
        if not self._can_draw(): return False
        return x in range(self.map.footer.width) and y in range(self.map.footer.height)

    def _get_tile(self, id, pal, hflip, vflip):
        """ Retrieves a tile from the tile pool """
        if not self.map.footer.tsp or not self.map.footer.tss: return
        if pal < 7: colors = self.map.footer.tsp.palettes[pal]
        else: colors = self.map.footer.tss.palettes[pal - 7]
        if id < 0x280: return self.map.footer.tsp.get_tile(id, colors, hflip, vflip)
        else: return self.map.footer.tss.get_tile(id - 0x280, colors, hflip, vflip)

    def _compute_block(self, primary, id):
        """ Computes a block in a certain tileset """
        if not self.map.footer.tsp or not self.map.footer.tss: return
        t = self.map.footer.tsp if primary else self.map.footer.tss
        block = t.blocks[id]
        return self._compute_block_by_data(block)

    def _compute_block_by_data(self, data):
        """ Computes a block by its tilemap data"""
        img = PImage.new("RGBA", (16, 16), "black")
        for k in range(8):
            map_data = data[k]
            id, vflip, hflip = map_data & 0x3FF, bool(map_data & 0x400), bool(map_data & 0x800)
            pal_id = (map_data >> 12) & 0xF
            if pal_id > 12: continue
            j = k & 3 #Index of inner loop (the computing loop is 2 layered, j runs from 0 to 3 for every layer)
            x, y = j & 1, j >> 1
            tile_image = self._get_tile(id, pal_id, hflip, vflip)
            if k < 4: #Paste without alpha channel on color #0
                img.paste(tile_image, (8 * x, 8 * y))
            else: #Paste with alpha channel on color #0
                img.paste(tile_image, (8 * x, 8 * y), tile_image)
        return img


    def _refresh_blocks(self):
        """ Refreshes all blocks """
        if not self._can_draw(): return self.block_canvas.delete("all")
        if not self.map.footer.tsp or not self.map.footer.tss: return
        tsp, tss = self.map.footer.tsp, self.map.footer.tss
        #First we create a pool of all blocks
        def get_tile(id, hflip, vflip, pal):
            #Inner method to retrieve a tile from the tile pool in a certain color
            colors = tsp.palettes[i].colors if pal < 7 else tss.palettes[i - 7].colors
            return tsp.get_tile(id, colors, hflip, vflip) if tile < 0x280 else tss.get_tile(id - 0x280, colors, hflip, vflip)
        self.block_imgs = [self._compute_block(True, i) for i in range(len(self.map.footer.tsp.blocks))] + [self._compute_block(False, i) for i in range(len(self.map.footer.tss.blocks))]
        self.block_raw_pil_image = PImage.new("RGBA", (128, 2048), "black")
        for i in range(len(self.block_imgs)):
            x, y = i & 7, i >> 3
            self.block_raw_pil_image.paste(self.block_imgs[i], (16 * x, 16 * y))
        self.block_canvas.delete(self.block_canvas_image)
        self.block_raw_image = ImageTk.PhotoImage(self.block_raw_pil_image)
        self.block_canvas_image = self.block_canvas.create_image(1, 1, image=self.block_raw_image, anchor=tkinter.NW)


    def _refresh_borders(self):
        """ Refresh the borders """
        if not self._can_draw(): return self.border_canvas.delete("all")
        bw, bh = self.map.footer.border_width, self.map.footer.border_height
        self.border_canvas.grid_remove()
        self.border_canvas = tkinter.Canvas(self.block_widget_border_pane, width=16 * bw, height=16* bh, background="black")
        self.border_canvas.grid(row=0, column=0, sticky=tkinter.NW)
        self.border_raw_pil_image = PImage.new("RGBA", (bw * 16, bh * 16), "black")
        for y in range(len(self.map.footer.borders)):
            for x in range(len(self.map.footer.borders[y])):
                block_id, level = mapheader.block_data(self.map.footer.borders[y][x])
                img = self.block_imgs[block_id]
                self.border_raw_pil_image.paste(img, (16 * x, 16 * y))
        self.border_raw_image = ImageTk.PhotoImage(self.border_raw_pil_image)
        self.border_canvas.create_image(1, 1, image=self.border_raw_image, anchor=tkinter.NW)

        #Event binding of border canvas
        def border_canvas_b_pressed(e):
            x, y = int(e.x / 16), int(e.y / 16)
            self.action(Action_border_change(self, x, y, self.block_selection))
            self.border_canvas_last_coords = x, y

        def border_canvas_b_motion(e):
            x, y = int(e.x / 16), int(e.y / 16)
            xl, yl = self.border_canvas_last_coords
            if x != xl or y != yl:
                self.action(Action_border_change(self, x, y, self.block_selection))
                self.border_canvas_last_coords = x, y

        def border_canvas_select(e):
            x, y = int(e.x / 16), int(e.y / 16)
            block = self.map.footer.borders[y][x]
            bx, by = block % 8, int(block / 8)
            self.update_block_selection_by_block_area(bx, by, bx + 1, by + 1)
            #Adjust the block canvas to the upper left block
            slider_top = by / 128
            relative_canvas_height = BLOCK_CANVAS_SIZE[HEIGHT] / 2048
            if slider_top > 1 - relative_canvas_height: slider_top = 1 - relative_canvas_height
            slider_bottom = slider_top + relative_canvas_height
            self.block_canvas_bar_v.set(slider_top, slider_bottom)
            self.block_canvas.yview("moveto", slider_top)
                
        self.border_canvas.bind("<B1-Motion>", border_canvas_b_motion)
        self.border_canvas.bind("<ButtonPress-1>", border_canvas_b_pressed)
        self.border_canvas.bind("<ButtonPress-3>", border_canvas_select)

    def _refresh_block_selection(self):
        """ Refresh the selection widget"""
        if not self._can_draw(): return self.selection_canvas.delete("all")
        _selection_frame = self.selection_frame
        ow, oh = self.block_selection_size
        w, h = len(self.block_selection[0]), len(self.block_selection)
        if ow != w or oh != h:
            #Create a new frame and canvas
            cw, ch = min(w, 8), min(h, 4)
            self.selection_frame = tkinter.Frame(self.block_widget_selection_pane)
            self.selection_frame.grid(row=1, column=0, sticky=tkinter.NSEW)
            self.selection_canvas = tkinter.Canvas(self.selection_frame, width = 16 * cw, height = 16 * ch, background="black", scrollregion=(0, 0, 16 * w, 16 * h))
            if w > cw or h > ch:
                if h > ch:
                    #Add v scrollbar
                    self.selection_canvas_bar_v = tkinter.Scrollbar(self.selection_frame, orient=tkinter.VERTICAL)
                    self.selection_canvas_bar_v.pack(side=tkinter.RIGHT, fill=tkinter.Y)
                    self.selection_canvas_bar_v.config(command=self.selection_canvas.yview)
                    self.selection_canvas.config(yscrollcommand=self.selection_canvas_bar_v.set)
                if w > cw:
                    #Add h scrollbar
                    self.selection_canvas_bar_h = tkinter.Scrollbar(self.selection_frame, orient=tkinter.HORIZONTAL)
                    self.selection_canvas_bar_h.pack(side=tkinter.BOTTOM, fill=tkinter.X)
                    self.selection_canvas_bar_h.config(command=self.selection_canvas.xview)
                    self.selection_canvas.config(xscrollcommand=self.selection_canvas_bar_h.set)
            self.selection_canvas.pack(side=tkinter.LEFT)
            _selection_frame.grid_remove()
        self.block_selection_size = w, h

        self.selection_raw_pil_image = PImage.new("RGBA", (w * 16, h * 16), "black")
        for y in range(len(self.block_selection)):
            for x in range(len(self.block_selection[y])):
                block_id = self.block_selection[y][x]
                self.selection_raw_pil_image.paste(self.block_imgs[block_id], (16 * x, 16 * y))
        self.selection_raw_image = ImageTk.PhotoImage(self.selection_raw_pil_image)
        self.selection_canvas.create_image(1, 1, image=self.selection_raw_image, anchor=tkinter.NW)
        


    def undo(self):
        """ Pops from undo stack and undoes last action """
        if len(self.undo_stack):
            action = self.undo_stack.pop()
            self.redo_stack.append(action)
            action.undo()

    def redo(self):
        """ Pops from redo stack and redoes the last action """
        if len(self.redo_stack):
            action = self.redo_stack.pop()
            self.undo_stack.append(action)
            action.do()

    


    def _refresh_map(self):
        """ Refresh the map """
        if not self._can_draw(): return self.map_widget_canvas.delete("all")
        self._compute_map_image()
        self._sync_map_canvas_with_map_img()

    def color_by_level(self, level):
        """ Returns a unique color for each height """
        if level == 0: return "blue"
        if level == 1: return "red"
        if level == 4: return "magenta"
        if level == 0xC: return "#800080"
        rgb = [0, 0, 0]
        for i in range(6):
            rgb[i % 3] |= (level & 1) << (int(i/3))
            level >>= 1
        r, g, b = rgb[0] << 2, rgb[1] << 2, rgb[2] << 2
        return "#" + hex(r)[2:] + hex(g)[2:] + hex(b)[2:]

    def _refresh_level_map(self):
        """ Refreshes the level map """
        if not self._can_draw(): return self.level_widget_canvas.delete("all")
        self._compute_map_image()
        self.level_widget_canvas.config(scrollregion = (0, 0, 16 * self.map.footer.width, 16 * self.map.footer.height))
        self.level_widget_canvas.delete("all")
        self.level_widget_canvas_map_image_raw = ImageTk.PhotoImage(self.map_img)
        self.level_widget_canvas_map_image = self.level_widget_canvas.create_image(1, 1, image=self.level_widget_canvas_map_image_raw, anchor=tkinter.NW) #Use the raw image of the map widget
        self.level_widget_canvas_objects = [[None for j in range(self.map.footer.width)] for i in range(self.map.footer.height)]
        for y in range(self.map.footer.height):
            for x in range(self.map.footer.width):
                level = self.map.footer.blocks[y][x] >> 10
                rect = self.level_widget_canvas.create_rectangle(16 * x, 16 * y, 16 * (x + 1), 16 * (y + 1), fill=self.color_by_level(level), stipple="gray50")
                text = self.level_widget_canvas.create_text(16 * x + 8, 16 * y + 8, text=_strcap(hex(level)[2:]), anchor=tkinter.CENTER, justify=tkinter.CENTER, width=16)
                self.level_widget_canvas_objects[y][x] = text, rect

    def _refresh_level_map_coordinate(self, x, y):
        """ Refreshes the level map at one coordinate only """
        if not self._can_draw(): return
        text, rect = self.level_widget_canvas_objects[y][x]
        level = self.map.footer.blocks[y][x] >> 10
        self.level_widget_canvas.itemconfig(text, text=_strcap(hex(level)[2:]))
        self.level_widget_canvas.itemconfig(rect, fill=self.color_by_level(level))
            
    def _refresh_event_person_picture(self, id):
        """ Refreshes the picture of a person event"""
        if not self._can_draw() or not self.show_ow_imgs: return
        person = self.map.persons[id] 
        x, y = person.x, person.y
        img, pimg = self.ow_img_pool.get(person.picture)
        width, height = img.size
        cimg, _, _ = self.event_map_widget_canvas_person_objects[id]
        self.event_map_widget_canvas.delete(cimg)
        cimg = self.event_map_widget_canvas.create_image(x * 16 - (width >> 1) + 8, y * 16 - (height) + 16, image=pimg, anchor=tkinter.NW)
        self.event_map_widget_canvas_person_objects[id] = ((cimg, width, height))



    def _refresh_event_map(self):
        """ Refreshes the event map """
        if not self._can_draw(): return self.event_map_widget_canvas.delete("all")
        self._compute_map_image()
        self.event_map_widget_canvas.config(scrollregion = (0, 0, 16 * self.map.footer.width, 16 * self.map.footer.height))
        self.event_map_widget_canvas.delete("all")
        self.event_map_widget_canvas_map_image_raw = ImageTk.PhotoImage(self.map_img)
        self.event_map_widget_canvas_map_image = self.event_map_widget_canvas.create_image(1, 1, image=self.event_map_widget_canvas_map_image_raw, anchor=tkinter.NW) #Use the raw image of the map
        #Draw each person
        self.event_map_widget_canvas_person_objects = []
        for person in self.map.persons:
            x, y = person.x, person.y
            if self.show_ow_imgs:
                img, pimg = self.ow_img_pool.get(person.picture)
                width, height = img.size
                cimg = self.event_map_widget_canvas.create_image(x * 16 - (width >> 1) + 8, y * 16 - (height) + 16, image=pimg, anchor=tkinter.NW)
                self.event_map_widget_canvas_person_objects.append((cimg, width, height))
            else:
                rect = self.event_map_widget_canvas.create_rectangle(16 * x, 16 * y, 16 * (x + 1), 16 * (y + 1), fill="#00FF00", stipple="gray50")
                text = self.event_map_widget_canvas.create_text(16 * x + 8, 16 * y + 8, text="P", anchor=tkinter.CENTER, justify=tkinter.CENTER, width=16)
                self.event_map_widget_canvas_person_objects.append((text, rect))
        #Draw each warp
        self.event_map_widget_canvas_warp_objects = []
        for warp in self.map.warps:
            x, y = warp.x, warp.y
            rect = self.event_map_widget_canvas.create_rectangle(16 * x, 16 * y, 16 * (x + 1), 16 * (y + 1), fill="#FF00FF", stipple="gray50")
            text = self.event_map_widget_canvas.create_text(16 * x + 8, 16 * y + 8, text="W", anchor=tkinter.CENTER, justify=tkinter.CENTER, width=16)
            self.event_map_widget_canvas_warp_objects.append((text, rect))
        #Draw each trigger
        self.event_map_widget_canvas_trigger_objects = []
        for trigger in self.map.triggers:
            x, y = trigger.x, trigger.y
            rect = self.event_map_widget_canvas.create_rectangle(16 * x, 16 * y, 16 * (x + 1), 16 * (y + 1), fill="#0000FF", stipple="gray50")
            text = self.event_map_widget_canvas.create_text(16 * x + 8, 16 * y + 8, text="T", anchor=tkinter.CENTER, justify=tkinter.CENTER, width=16)
            self.event_map_widget_canvas_trigger_objects.append((text, rect))
        #Draw each sign
        self.event_map_widget_canvas_sign_objects = []
        for sign in self.map.signposts:
            x, y = sign.x, sign.y
            rect = self.event_map_widget_canvas.create_rectangle(16 * x, 16 * y, 16 * (x + 1), 16 * (y + 1), fill="#FF0000", stipple="gray50")
            text = self.event_map_widget_canvas.create_text(16 * x + 8, 16 * y + 8, text="S", anchor=tkinter.CENTER, justify=tkinter.CENTER, width=16)
            self.event_map_widget_canvas_sign_objects.append((text, rect))
        
        self._refresh_event_edit_active_rectangle()

    def _refresh_event_edit_active_rectangle(self):
        """ Refreshes the red active box arround the active event """
        #Draw red box arround selected event
        if self.event_map_widget_canvas_selection_rectangles: #We have to use more than one rectangle to make the outline bigger
            for rect in self.event_map_widget_canvas_selection_rectangles:
                self.event_map_widget_canvas.delete(rect)
            self.event_map_widget_canvas_selection_rectangles = None
        if self.event_id == None: return
        try:
            if self.event_type == "Person": event = self.map.persons[self.event_id]
            elif self.event_type == "Warp": event = self.map.warps[self.event_id]
            elif self.event_type == "Trigger": event = self.map.triggers[self.event_id]
            elif self.event_type in ("Sign", "SignScript", "SignItem"): event = self.map.signposts[self.event_id]
            else: return
        except IndexError: return #There is not rectangle anymore since self.event_id is not a valid id anymore
        x, y = event.x, event.y
        self.event_map_widget_canvas_selection_rectangles = [self.event_map_widget_canvas.create_rectangle(16 * x + i, 16 * y + i, 16 * (x + 1) - i, 16 * (y + 1) - i, outline="red") for i in range(-1, 1)]

    def _sync_map_canvas_with_map_img(self):
        """ Syncronizes the map canvas with self.map_img """
        if not self._can_draw(): return
        self.map_widget_canvas.config(scrollregion = (0, 0, 16 * self.map.footer.width, 16 * self.map.footer.height))
        self.raw_map_img = ImageTk.PhotoImage(self.map_img)
        if self.map_widget_canvas_image: self.map_widget_canvas.delete(self.map_widget_canvas_image)
        self.map_widget_canvas_image = self.map_widget_canvas.create_image(1, 1, image=self.raw_map_img, anchor=tkinter.NW)
    
    def _refresh_map_block(self, x, y):
        """ Refreshes a single block on the map"""
        if not self._can_draw(): return
        block = self.map.footer.blocks[y][x] & 0x3FF
        self.map_img.paste(self.block_imgs[block], (x * 16, y * 16))
        self._sync_map_canvas_with_map_img()
    
    def _reassing_tilesets(self):
        """ Tries to reassign tilesets to the map """
        if not self.map: return
        tsp_sym = self.map.footer.tsp_sym
        if tsp_sym:
            self.map.footer.tsp = self.proj.get_tileset(tsp_sym)
        tss_sym = self.map.footer.tss_sym
        if tss_sym:
            self.map.footer.tss = self.proj.get_tileset(tss_sym)
        

    def _refresh_title(self):
        """ Refreshes the title """
        if self.map: self.root.wm_title("Pymap - " + self.map.key)
        else: self.root.wm_title("Pymap - no map loaded")

    def _refresh(self):
        """ Refreshes all widgets """
        self._refresh_title()
        self._reassing_tilesets()
        self._refresh_blocks()
        self._refresh_borders()
        self._refresh_block_selection()
        self._refresh_map()
        self._refresh_level_map()
        self._refresh_event_map()
        #Clear current event:
        self.event_edit_type_dropdown.set("None")
        self._refresh_event_edit_active_rectangle()

    def _refresh_context_map(self):
        """ Refreshes only the current visible map and expects the other maps to sync when called (lazy but fast) """
        if self.context == MAP: self._refresh_map()
        elif self.context == LEVEL: self._refresh_level_map()
        elif self.context == EVENTS: self._refresh_event_map()

    def map_has_changed(self):
        """ Checks if the current map (if present) has changed and thus should be saved """
        if not self.map: return False
        last_action = self.action_peek()
        if not last_action: return False
        if last_action.is_last_saved_state: return False
        return True

    def open_map(self, bank, mapid):
        """ Opens a map by its bank and map"""
        
        if self.map and self.map_has_changed():
            if messagebox.askyesno(title="Save changes", message="Do you want to save your changes to map " + str(self.bank) + "." + str(self.mapid) + "?"): self.file_save()
        path = self.proj.get_map_path(bank, mapid)
        if not path or not len(path): return
        try: self.map = mapheader.load(path, self.proj)
        except Exception as e:
            messagebox.showerror(title="Unable to open map", message="Unable to open map " + str(bank) + "." + str(id) + " at '" + path + "': "  + str(e))
            return
        self.bank, self.mapid = bank, mapid
        self._refresh()
        self.undo_stack = [] #No memory for a newly opened map
        self.redo_stack = [] #as above
        print("Map file " + path + " (" + str(bank) + "." + str(mapid) + ") opened successfully")
    

    def file_save(self):
        """ Saves a file """
        if not self.map: return
        self.proj.save_map(self.bank, self.mapid, self.map, self.proj.get_map_path(self.bank, self.mapid))
        self.map_tree.refresh()
        self.proj.save_project()
        #Mark the topmost element of the undo stack as last saved state
        for i in range(len(self.undo_stack)):
            if i < len(self.undo_stack) - 1: self.undo_stack[i].is_last_saved_state = False
            else: self.undo_stack[i].is_last_saved_state = True
        for i in range(len(self.redo_stack)): self.redo_stack[i].is_last_saved_state = False

        print("Map file " + str(self.bank) + "." + str(self.mapid) + " saved sucessfully")
        return True


    def edit_header(self):
        """ Opens a dialog for header editing """
        if not self.map: return
        if self.header_dialog:
            self.header_dialog.focus_force()
            return
        dialog = tkinter.Toplevel(self.root)
        self.header_dialog = dialog
        dialog.wm_title("Edit mapheader of map " + self.map.key)
        dialog.attributes("-toolwindow", 1)
        entries = [("Levelscript header", "levelscript_header", None), ("Music", "music", constants.music),
        ("Flash type", "flash_type", constants.flash_types), ("Weather", "weather", constants.map_weathers), ("Map type", "type", constants.map_types),
        ("Field18", "field_18", None), ("Show name", "show_name", constants.map_show_name), ("Field1A", "field_1a", None) , ("Battle style", "battle_style", constants.battle_types),
        ("Footer id", "id", None)
        ]
        dialog.dropdowns = [None for _ in entries]
        for i in range(len(entries)):
            show, name, consts = entries[i]
            tkinter.Label(dialog, text=show + ":").grid(row=i, column=0, sticky=tkinter.NW)
            if consts: entry = ttk.Combobox(dialog, values=list(constants.values(consts)))
            else: entry = tkinter.Entry(dialog)
            entry.grid(row=i, column=1, sticky=tkinter.NW)
            value = self.map.__getattribute__(name)
            try: value = hex(value)
            except: value = str(value)
            entry.insert(0, value)
            entries[i] = show, name, entry
                            

        def close():
            #self.root.deiconify()
            dialog.destroy()
            self.header_dialog = None
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.entries = entries

        def update():
            """ Initializes the entries with values"""
            if not self.header_dialog: return
            for i in range(len(self.header_dialog.entries)):
                show, name, entry = self.header_dialog.entries[i]
                value = self.map.__getattribute__(name)
                try: value = hex(value)
                except: value = str(value)
                entry.delete(0, tkinter.END)
                entry.insert(0, value)
        update()


        def apply_changes():
            """ Applies changes by using a action class """
            diffs = {}
            for i in range(len(self.header_dialog.entries)):
                show, name, entry = self.header_dialog.entries[i]
                value = entry.get()
                try: value = int(value, 0)
                except: pass
                value_old = self.map.__getattribute__(name)
                if value != value_old: diffs[name] = value, value_old
            if "id" in diffs:
                #Check if the new footer id is already used (if another map uses this footer id)
                footer_id, _ = diffs["id"]
                id_usage = self.proj.get_footer_usage(footer_id)
                print(diffs["id"], id_usage)
                if self.map.symbol in id_usage: id_usage.remove(self.map.symbol)
                if len(id_usage):
                    if not tkinter.messagebox.askyesno(title="Footer id already used", message="The footer id " + str(footer_id) + " is already used by the following map(s) (listed by symbol) "+ str(id_usage) + ". A possible unused id would be " + hex(self.proj.get_smallest_availible_foooter_id()) + ". Do you want to continue?", default="no"): return


            if len(diffs): self.action(Action_header_edit(self, diffs, update))

        tkinter.Button(dialog, text="Apply", command=apply_changes).grid(row=len(entries), column=0, sticky=tkinter.NW)
                
    
    

    def edit_connections(self):
        if not self.map: return
        #Try to force to an open dialog first
        try:
            self.connection_dialog.focus_force()
            return
        except: pass
        dialog = tkinter.Toplevel(self.root)
        self.connection_dialog = dialog
        dialog.wm_title("Edit connections of map " + self.map.key)
        dialog.attributes("-toolwindow", 1)
        dialog.current_connection_num = None

        

        def dialog_update():
            """ Updates the dialog """
            self.connection_dialog.connection_ids.config(values=list(map(str, range(len(self.map.connections)))))
            cur_id = self.connection_dialog.connection_ids.get()
            if cur_id == "" or cur_id == None:
                for w in (self.connection_dialog.direction, self.connection_dialog.displacement, self.connection_dialog.bank, self.connection_dialog.mapid, self.connection_dialog.fieldA, self.connection_dialog.fieldB):
                    w.set("", untraced=True)
                return
            cur_id = int(cur_id)
            if cur_id not in range(len(self.map.connections)):
                self.connection_dialog.connection_ids.set("", untraced=True)
                dialog_update()
                return
            load_connection()

        def load_connection():
            """ Loads a connection"""
            if not self.map: return self.connection_dialog.destroy()
            self.connection_dialog.current_connection_num = int(self.connection_dialog.connection_ids.get())
            self.connection_dialog.connection = self.map.connections[self.connection_dialog.current_connection_num]
            self.connection_dialog.direction.set(str(self.connection_dialog.connection.direction), untraced=True)
            self.connection_dialog.displacement.set(str(self.connection_dialog.connection.displacement), untraced=True)
            self.connection_dialog.bank.set(str(self.connection_dialog.connection.bank), untraced=True)
            self.connection_dialog.mapid.set(str(self.connection_dialog.connection.mapid), untraced=True)
            self.connection_dialog.fieldA.set(str(self.connection_dialog.connection.fieldA), untraced=True)
            self.connection_dialog.fieldB.set(str(self.connection_dialog.connection.fieldB), untraced=True)
            update_image()

        def update_image():
            """ Updates the connection image """
            if not self.map: return dialog.destroy()
            if not self._can_draw(): return
            try:
                b = int(self.connection_dialog.bank.get())
                m = int(self.connection_dialog.mapid.get())
                self.connection_dialog.adjacent_map = mapheader.load(self.proj.get_map_path(b, m), self.proj)
                self.connection_dialog.adjacent_image = dialog.adjacent_map.get_map_image()
                self.connection_dialog.own_image = self.map_img
                if self.connection_dialog.adjacent_image == None or self.connection_dialog.own_image == None: return
                self.connection_dialog.own_pimage = ImageTk.PhotoImage(self.connection_dialog.own_image)
                self.connection_dialog.adjacent_pimage = ImageTk.PhotoImage(self.connection_dialog.adjacent_image)
                self.connection_dialog.canvas.delete("all") #clear canvas
                self.connection_dialog.own_cimage = self.connection_dialog.canvas.create_image(0, 0, image=self.connection_dialog.own_pimage, anchor=tkinter.NW)
                self.connection_dialog.adjacent_cimage = self.connection_dialog.canvas.create_image(0, 0, image=self.connection_dialog.adjacent_pimage, anchor=tkinter.NW)
                self.connection_dialog.border_line = self.connection_dialog.canvas.create_line(0, 0, 1, 1, fill="red")
                rearrange_maps()
            except Exception as e:
                print("Warning", e)

        def rearrange_maps():
            """ Rearranges the maps according to their adjacance"""
            if not self.map: return dialog.destroy()
            try:
                d = int(self.connection_dialog.displacement.get())
                adjacence_type = mapconnection.connection_type_to_adjacence(self.connection_dialog.direction.get())
                if adjacence_type in (mapconnection.NORTH, mapconnection.SOUTH):
                    if adjacence_type == mapconnection.NORTH: ay, oy = 0, self.connection_dialog.adjacent_map.footer.height
                    elif adjacence_type == mapconnection.SOUTH: ay, oy = self.map.footer.height, 0
                    width = (abs(d) + max(self.map.footer.width, self.connection_dialog.adjacent_map.footer.width))
                    height = self.map.footer.height + self.connection_dialog.adjacent_map.footer.height
                    self.connection_dialog.canvas.config(scrollregion = (0, 0, 16 * width, 16 * height))
                    ax = d if d > 0 else 0
                    ox = -d if d < 0 else 0
                    self.connection_dialog.canvas.delete(self.connection_dialog.border_line)
                    self.connection_dialog.border_line = self.connection_dialog.canvas.create_line(0, (ay + oy) * 16, width * 16, (ay + oy) * 16, fill="red")
                elif adjacence_type in (mapconnection.EAST, mapconnection.WEST):
                    if adjacence_type == mapconnection.EAST: ax, ox = self.map.footer.width, 0
                    elif adjacence_type == mapconnection.WEST: ax, ox = 0, self.connection_dialog.adjacent_map.footer.width
                    height = (abs(d) + max(self.map.footer.height, self.connection_dialog.adjacent_map.footer.height))
                    width = self.map.footer.width + self.connection_dialog.adjacent_map.footer.width
                    self.connection_dialog.canvas.config(scrollregion = (0, 0, 16 * width, 16 * height))
                    ay = d if d > 0 else 0
                    oy = -d if d < 0 else 0
                    self.connection_dialog.canvas.delete(self.connection_dialog.border_line)
                    self.connection_dialog.border_line = self.connection_dialog.canvas.create_line((ax + ox) * 16, 0, (ax + ox) * 16, height * 16, fill="red")
                elif adjacence_type == mapconnection.OVER:
                    ax, ox, ay, oy = 0, 0, 0, 0 #All overlay
                    self.connection_dialog.canvas.delete(self.connection_dialog.border_line)
                    self.connection_dialog.border_line = self.connection_dialog.canvas.create_line(0, 0, 0, 0, fill="red")
                else: return

                self.connection_dialog.canvas.coords(dialog.own_cimage, ox * 16, oy * 16)
                self.connection_dialog.canvas.coords(dialog.adjacent_cimage, ax * 16, ay * 16)

            except Exception as e:
                print("Warning", e)

        def add():
            """ Adds a new connection """
            if not self.map: dialog.destroy()
            self.action(Action_add_connection(self, dialog_update))

        def remove():
            """ Removes the current connection """
            if not self.map: dialog.destroy()
            cur_id = self.connection_dialog.connection_ids.get()
            if cur_id == "" or cur_id == None: return #No connection selected, so you can not remove it I guess
            self.action(Action_remove_connection(self, dialog_update, int(cur_id)))

        def apply():
            """ Applies changes to the current connection """
            if not self.map: dialog.destroy()
            def to_int(i):
                try: return int(i)
                except: return i
            id = self.connection_dialog.connection_ids.get()
            if id == None or id == "": return
            connection = self.map.connections[int(id)]
            adjance_type_new = to_int(self.connection_dialog.direction.get())
            displacement_new = to_int(self.connection_dialog.displacement.get())
            bank_new = to_int(self.connection_dialog.bank.get())
            mapid_new = to_int(self.connection_dialog.mapid.get())
            fieldA_new = to_int(self.connection_dialog.fieldA.get())
            fieldB_new = to_int(self.connection_dialog.fieldB.get())
            diffs = {}
            if adjance_type_new != connection.direction: diffs["direction"] = adjance_type_new, connection.direction
            if displacement_new != connection.displacement: diffs["displacement"] = displacement_new, connection.displacement
            if bank_new != connection.bank: diffs["bank"] = bank_new, connection.bank
            if mapid_new != connection.mapid: diffs["mapid"] = mapid_new, connection.mapid
            if fieldA_new != connection.fieldA: diffs["fieldA"] = fieldA_new, connection.fieldA
            if fieldB_new != connection.fieldB: diffs["fieldB"] = fieldB_new, connection.fieldB
            if not len(diffs): return #No changes...
            self.action(Action_connection_change(self, dialog_update, int(id), diffs))



        self.connection_dialog.widget_frame = tkinter.Frame(dialog)
        self.connection_dialog.widget_frame.grid(row=0, column=0, sticky=tkinter.NW)
        self.connection_dialog.connection_ids = tkinterx.LabeledCombobox(self.connection_dialog.widget_frame, list(map(str, range(len(self.map.connections)))), text="Connection #", callback=load_connection)
        self.connection_dialog.connection_ids.config(state="readonly")
        self.connection_dialog.connection_ids.get_handle().grid(row=0, column=0, sticky=tkinter.NW)
        tkinter.Label(self.connection_dialog.widget_frame, text="Adjacence type").grid(row=1, column=0, sticky=tkinter.NW)
        self.connection_dialog.direction = tkinterx.Combobox(self.connection_dialog.widget_frame, constants.map_connections, callback=rearrange_maps)
        self.connection_dialog.direction.grid(row=1, column=1, sticky=tkinter.NW)
        tkinter.Label(self.connection_dialog.widget_frame, text="Displacement").grid(row=2, column=0, sticky=tkinter.NW)
        self.connection_dialog.displacement = tkinterx.Entry(self.connection_dialog.widget_frame, callback=rearrange_maps)
        self.connection_dialog.displacement.grid(row=2, column=1, sticky=tkinter.NW)
        tkinter.Label(self.connection_dialog.widget_frame, text="Target bank").grid(row=3, column=0, sticky=tkinter.NW)
        self.connection_dialog.bank = tkinterx.Entry(self.connection_dialog.widget_frame, callback=update_image)
        self.connection_dialog.bank.grid(row=3, column=1, sticky=tkinter.NW)
        tkinter.Label(self.connection_dialog.widget_frame, text="Target map").grid(row=4, column=0, sticky=tkinter.NW)
        self.connection_dialog.mapid = tkinterx.Entry(self.connection_dialog.widget_frame, callback=update_image)
        self.connection_dialog.mapid.grid(row=4, column=1, sticky=tkinter.NW)
        tkinter.Label(self.connection_dialog.widget_frame, text="FieldA").grid(row=5, column=0, sticky=tkinter.NW)
        self.connection_dialog.fieldA = tkinterx.Entry(self.connection_dialog.widget_frame)
        self.connection_dialog.fieldA.grid(row=5, column=1, sticky=tkinter.NW)
        tkinter.Label(self.connection_dialog.widget_frame, text="FieldB").grid(row=6, column=0, sticky=tkinter.NW)
        self.connection_dialog.fieldB = tkinterx.Entry(self.connection_dialog.widget_frame)
        self.connection_dialog.fieldB.grid(row=6, column=1, sticky=tkinter.NW)
        self.connection_dialog.canvas_frame = tkinter.Frame(self.connection_dialog)
        self.connection_dialog.canvas_frame.grid(row=0, column=1, sticky=tkinter.NW)
        self.connection_dialog.canvas = tkinter.Canvas(self.connection_dialog.canvas_frame, width=int(MAP_CANVAS_SIZE[WIDTH] * 0.5), height=int(MAP_CANVAS_SIZE[HEIGHT] * 0.5), background="lightgray")
        self.connection_dialog.canvas.grid(row=0, column=0, sticky=tkinter.NW)
        self.connection_dialog.canvas_bar_v = tkinter.Scrollbar(self.connection_dialog.canvas_frame, orient=tkinter.VERTICAL)
        self.connection_dialog.canvas_bar_v.grid(row=0, column=1, sticky=tkinter.NSEW)
        self.connection_dialog.canvas_bar_v.config(command=self.connection_dialog.canvas.yview)
        self.connection_dialog.canvas.config(yscrollcommand=self.connection_dialog.canvas_bar_v.set)
        self.connection_dialog.canvas_bar_h = tkinter.Scrollbar(self.connection_dialog.canvas_frame, orient=tkinter.HORIZONTAL)
        self.connection_dialog.canvas_bar_h.grid(row=1, column=0, sticky=tkinter.NSEW)
        self.connection_dialog.canvas_bar_h.config(command=self.connection_dialog.canvas.xview)
        self.connection_dialog.canvas.config(xscrollcommand=self.connection_dialog.canvas_bar_h.set)
        self.connection_dialog.bt_add = tkinter.Button(self.connection_dialog.widget_frame, text="Add", command=add)
        self.connection_dialog.bt_add.grid(row=7, column=0, sticky=tkinter.NW)
        self.connection_dialog.bt_remove = tkinter.Button(self.connection_dialog.widget_frame, text="Remove", command=remove)
        self.connection_dialog.bt_remove.grid(row=8, column=0, sticky=tkinter.NW)
        self.connection_dialog.bt_apply = tkinter.Button(self.connection_dialog.widget_frame, text="Apply", command=apply)
        self.connection_dialog.bt_apply.grid(row=9, column=0, sticky=tkinter.NW)



        def close():
            #self.root.deiconify()
            dialog.destroy()
            self.connection_dialog = None
        dialog.protocol("WM_DELETE_WINDOW", close)


    def edit_footer(self):
        """ Opens a dialog for footer editing """
        if not self.map: return
        try:
            self.footer_dialog.focus_force()
            return
        except: pass
        dialog = tkinter.Toplevel(self.root)
        self.footer_dialog = dialog
        #self.root.withdraw()
        dialog.wm_title("Edit mapfooter of map " + self.map.key)
        dialog.attributes("-toolwindow", 1)
        tkinter.Label(dialog, text="Primary tileset:").grid(row=0, column=0, sticky=tkinter.NW)
        tkinter.Label(dialog, text="Secondary tileset:").grid(row=1, column=0, sticky=tkinter.NW)
        entry_tsp, entry_tss = tkinterx.Combobox(dialog, values=self.proj.get_tileset_paths()), tkinterx.Combobox(dialog, values=self.proj.get_tileset_paths())
        entry_tsp.grid(row=0, column=1, sticky=tkinter.NW), entry_tss.grid(row=1, column=1, sticky=tkinter.NW)
        tkinter.Label(dialog, text="Map width:").grid(row=2, column=0, sticky=tkinter.NW)
        tkinter.Label(dialog, text="Map height:").grid(row=3, column=0, sticky=tkinter.NW)
        entry_width, entry_height = tkinter.Entry(dialog), tkinter.Entry(dialog)
        entry_width.grid(row=2, column=1, sticky=tkinter.NW), entry_height.grid(row=3, column=1, sticky=tkinter.NW)
        tkinter.Label(dialog, text="Border width:").grid(row=4, column=0, sticky=tkinter.NW)
        tkinter.Label(dialog, text="Border height:").grid(row=5, column=0, sticky=tkinter.NW)
        entry_border_width, entry_border_height = tkinter.Entry(dialog), tkinter.Entry(dialog)
        entry_border_width.grid(row=4, column=1, sticky=tkinter.NW), entry_border_height.grid(row=5, column=1, sticky=tkinter.NW)
        #Bind all entries in a ugly way so that a redo/undo event knows never a direct reference to an entry but rather to an dialoge that remebers its entries (undos will then affect new dialoges)
        dialog._etsp = entry_tsp
        dialog._etss = entry_tss
        dialog._ew = entry_width
        dialog._eh = entry_height
        dialog._ebw = entry_border_width
        dialog._ebh = entry_border_height

        def update():
            if not self.footer_dialog: return
            self.footer_dialog._etsp.delete(0, tkinter.END)
            self.footer_dialog._etss.delete(0, tkinter.END)
            if self.map.footer.tsp: self.footer_dialog._etsp.set(self.map.footer.tsp.symbol)
            else: self.footer_dialog._etsp.set("")
            if self.map.footer.tss: self.footer_dialog._etss.set(self.map.footer.tss.symbol)
            else: self.footer_dialog._etss.set("")
            self.footer_dialog._ew.delete(0, tkinter.END), self.footer_dialog._eh.delete(0, tkinter.END)
            self.footer_dialog._ew.insert(0, str(self.map.footer.width)), self.footer_dialog._eh.insert(0, str(self.map.footer.height))
            self.footer_dialog._ebw.delete(0, tkinter.END), self.footer_dialog._ebh.delete(0, tkinter.END)
            self.footer_dialog._ebw.insert(0, str(self.map.footer.border_width)), self.footer_dialog._ebh.insert(0, str(self.map.footer.border_height))
        update()

        def apply_changes():
            try:
                diffs = {}
                width_old, height_old = self.map.footer.width, self.map.footer.height
                width_new, height_new = int(entry_width.get(), 0), int(entry_height.get(), 0)
                bwidth_old, bheight_old = self.map.footer.border_width, self.map.footer.border_height
                bwidth_new, bheight_new = int(entry_border_width.get(), 0), int(entry_border_height.get(), 0)
                tsp_old, tss_old = self.map.footer.tsp_sym, self.map.footer.tss_sym
                tsp_new, tss_new = entry_tsp.get(), entry_tss.get()
                if width_new != width_old or height_new != height_old: 
                    diffs["width"] = width_new, width_old
                    diffs["height"] = height_new, height_old
                if bwidth_new != bwidth_old or bheight_new != bheight_old: 
                    diffs["border_width"] = bwidth_new, bwidth_old
                    diffs["border_height"] = bheight_new, bheight_old
                if tsp_new != tsp_old: diffs["tsp"] = tsp_new, tsp_old
                if tss_new != tss_old: diffs["tss"] = tss_new, tss_old
                if len(diffs): self.action(Action_footer_edit(self, diffs, update))


                """self.map.footer.width, self.map.footer.height = int(entry_width.get(), 0), int(entry_height.get(), 0)
                self.map.footer.border_width, self.map.footer.border_height = int(entry_border_width.get(), 0), int(entry_border_height.get(), 0)
                #Extend or cut map and block data
                new_blocks = [[0 for i in range(self.map.footer.width)] for j in range(self.map.footer.height)]
                for y in range(min(self.map.footer.height, len(self.map.footer.blocks))):
                    for x in range(min(self.map.footer.width, len(self.map.footer.blocks[y]))):
                        new_blocks[y][x] = self.map.footer.blocks[y][x]
                new_borders = [[0 for i in range(self.map.footer.border_width)] for j in range(self.map.footer.border_height)]
                for y in range(min(self.map.footer.border_height, len(self.map.footer.borders))):
                    for x in range(min(self.map.footer.border_width, len(self.map.footer.borders[y]))):
                        new_borders[y][x] = self.map.footer.borders[y][x]
                self.map.footer.blocks = new_blocks
                self.map.footer.borders = new_borders
                if len(entry_tsp.get()): self.map.footer.tsp = tileset.from_file(entry_tsp.get())
                if len(entry_tss.get()): self.map.footer.tss = tileset.from_file(entry_tss.get())
                self._refresh()"""
            except Exception as e:
                messagebox.showerror("Footer error", "Unable to save footer properties: " + str(e))
        tkinter.Button(dialog, text="Apply", command=apply_changes).grid(row=6, column=0, sticky=tkinter.NW)
        def close():
            #self.root.deiconify()
            dialog.destroy()
            self.footer_dialog = None
        dialog.protocol("WM_DELETE_WINDOW", close)

        #Binding for pyset
        def open_in_pyset():
            if not self.map.footer.tsp or not self.map.footer.tss: return
            tl = tkinter.Toplevel(self)
            ts_gui = tileset_gui.Tileset_gui(tl, self.proj)
            ts_gui.open_tileset(self.map.footer.tsp.symbol)
            ts_gui.open_tileset(self.map.footer.tss.symbol)
            tl.grab_set()
            self.wait_window(tl)
            self._reassing_tilesets()
            if self._can_draw(): self._refresh()
        tkinter.Button(dialog, text="Edit in Pyset...", command=open_in_pyset).grid(row=0, column=2, sticky=tkinter.NW)

    def display_ow_imgs(self):
        """ Opens a dialog and shows all image ows (for better selection) """
        if self.event_ow_image_widget: 
            self.event_ow_image_widget.grid_remove()
            self.event_ow_image_widget.destroy()
            self.event_ow_image_widget = None
            return
        self.event_ow_image_widget = ttk.Labelframe(self.event_frame, text="Imgs")
        self.event_ow_image_widget.grid(row=0, column=2, sticky=tkinter.NW)
        width = 64
        height = MAP_CANVAS_SIZE[HEIGHT]
        canvas = tkinter.Canvas(self.event_ow_image_widget, width=width, height=height, background="lightgray")
        canvas.grid(row=0, sticky=tkinter.NW)
        id_label = tkinter.Label(self.event_ow_image_widget, text="???")
        id_label.grid(row=1, sticky=tkinter.NW)
        
        #Now build up the picture
        self.event_ow_image_widget_image_origin = [0 for _ in range(256)]
        height = 0
        for i in range(256):
            ow_img, ow_pimg = self.ow_img_pool.get(i)
            self.event_ow_image_widget_image_origin[i] = height
            canvas.create_image(1, height + 1, image=ow_pimg, anchor=tkinter.NW)
            height += ow_img.size[1]
        canvas.config(scrollregion=(0, 0, 64, height))
        bar_v = tkinter.Scrollbar(self.event_ow_image_widget, orient=tkinter.VERTICAL)
        bar_v.grid(row=0, column=1, sticky=tkinter.NSEW)
        bar_v.config(command=canvas.yview)
        canvas.config(yscrollcommand=bar_v.set)

        def y_to_index(y):
            """ Uses binary search to find the index to a y coordinate """
            left = 0
            right = len(self.event_ow_image_widget_image_origin) - 1
            while(left <= right):
                middle = left + ((right - left) >> 1)
                if y == self.event_ow_image_widget_image_origin[middle]: return middle
                if y <= self.event_ow_image_widget_image_origin[middle]: right = middle - 1
                else: left = middle + 1
            return left - 1

        def canvas_motion(e):
            """ Motion over canvas """
            vslider, _ = bar_v.get()
            y = e.y + (vslider * height)
            i = y_to_index(int(y))
            id_label["text"] = "Pic " + hex(i)

        def canvas_b1_press(e):
            """ Click on canvas -> try to apply picture to current active person properties """
            vslider, _ = bar_v.get()
            y = e.y + (vslider * height)
            if self.context == EVENTS and self.event_type == "Person":
                try: 
                    entry_picture = self.__getattribute__("event_edit_widget_" + "Picture".lower())
                    i = y_to_index(int(y))
                    entry_picture.set(hex(i))
                except Exception as e: print(e)

        canvas.bind("<Motion>", canvas_motion)
        canvas.bind("<ButtonPress-1>", canvas_b1_press)

class Action:
    def __init__(self):
        self.is_last_saved_state = False
        

class Action_add_connection(Action):
    """ Action class for adding of connection """
    def __init__(self, root: Pymap_gui, dialog_update: callable):
        super().__init__()
        self.root = root
        self.id = len(root.map.connections)
        self.dialog_update = dialog_update
    
    def do(self):
        self.root.map.connections.append(mapconnection.Mapconnection())
        self.dialog_update()

    def undo(self):
        self.root.map.connections.remove(self.root.map.connections[self.id])
        self.dialog_update()

class Action_remove_connection(Action):
    """ Action class for removal of connection"""
    def __init__(self, root: Pymap_gui, dialog_update: callable, id):
        super().__init__()
        self.root = root
        self.id = id
        self.connection = root.map.connections[id]
        self.dialog_update = dialog_update
    
    def do(self):
        self.root.map.connections.remove(self.root.map.connections[self.id])
        self.dialog_update()

    def undo(self):
        self.root.map.connections.insert(self.id, self.connection)
        self.dialog_update()

class Action_connection_change(Action):
    """ Action class for editing of connection """
    def __init__(self, root: Pymap_gui, dialog_update: callable, id, diffs):
        super().__init__()
        self.root = root
        self.id = id
        self.connection = root.map.connections[id]
        self.dialog_update = dialog_update
        self.diffs = diffs
    
    def do(self):
        for k in self.diffs:
            n, o = self.diffs[k]
            self.root.map.connections[self.id].__setattr__(k, n)
        self.dialog_update()

    def undo(self):
        for k in self.diffs:
            n, o = self.diffs[k]
            self.root.map.connections[self.id].__setattr__(k, o)
        self.dialog_update()

class Action_event(Action):
    """ Abstract superclass for action events """
    def __init__(self, root: Pymap_gui, stype, id):
        super().__init__()
        self.root = root
        self.stype = stype
        self.id = id

    def _get_event(self):
        if self.stype == "Person": event = self.root.map.persons[self.id]
        elif self.stype == "Warp": event = self.root.map.warps[self.id]
        elif self.stype == "Trigger": event = self.root.map.triggers[self.id]
        elif self.stype in ("Sign", "SignScript", "SignItem"): event = self.root.map.signposts[self.id]
        else: raise Exception("Unknown event type " + stype)
        return event

    def _get_element_count(self, stype=None):
        if not stype:
            return sum(map(self._get_element_count, ("Person", "Warp", "Trigger", "Sign")))
        if self.stype == "Person": return len(self.root.map.persons)
        elif self.stype == "Warp": return len(self.root.map.warps)
        elif self.stype == "Trigger": return len(self.root.map.triggers)
        elif self.stype in ("Sign", "SignScript", "SignItem"): return len(self.root.map.signposts)
        else: raise Exception("Unknown event type " + stype)
        

    def _update(self):
        #stype may have changed for signs
        event = self._get_event()
        if self.stype in ("SignItem", "SignScript"):
            event._struct_setup()
            if event.structure == mapevent.SIGN_STRUCTURE_ITEM: stype = "SignItem"
            else: stype = "SignScript"
        #self.root._select_event(self.stype, self.id)
        self.root._refresh_event(self.stype, self.id)
        if self.root.event_type == self.stype and self.root.event_id == self.id:
            self.root._event_edit_sync_entries_with_active_event(self.stype, self.id)
        self.root._refresh_event_person_picture(self.id)
        #self.root._refresh_event_map()

class Action_header_edit(Action):
    """ Action class for change of header properties """
    def __init__(self, root: Pymap_gui, diffs, header_dialog_update: callable):
        super().__init__()
        self.root = root
        self.diffs = diffs
        self.header_dialog_update = header_dialog_update

    def do(self):
        for attribute in self.diffs:
            value_new, value_old = self.diffs[attribute]
            self.root.map.__setattr__(attribute, value_new)
        self.root._refresh()
        self.header_dialog_update()

    def undo(self):
        for attribute in self.diffs:
            value_new, value_old = self.diffs[attribute]
            self.root.map.__setattr__(attribute, value_old)
        self.root._refresh()
        self.header_dialog_update()
        

class Action_footer_edit(Action):
    """ Action class for change of footer properties """
    def __init__(self, root: Pymap_gui, diffs, footer_dialog_update: callable):
        super().__init__()
        self.root = root
        self.diffs = diffs
        self.footer_dialog_update = footer_dialog_update
        if "width" in diffs or "height" in diffs:
            self.map_blocks = root.map.footer.blocks[:]
        if "border_width" in diffs or "border_height" in diffs:
            self.border_blocks = root.map.footer.borders[:]
        

    def do(self):
        #Resize of map block
        if "width" in self.diffs and "height" in self.diffs:
            w_new, w_old = self.diffs["width"]
            h_new, h_old = self.diffs["height"]
            new_blocks = [[0 for i in range(w_new)] for j in range(h_new)]
            for y in range(min(h_new, h_old)):
                for x in range(min(w_new, w_old)):
                    new_blocks[y][x] = self.map_blocks[y][x]
            self.root.map.footer.blocks = new_blocks
            self.root.map.footer.width = w_new
            self.root.map.footer.height = h_new
        #Resize of border blocks
        if "border_width" in self.diffs and "border_height" in self.diffs:
            w_new, w_old = self.diffs["border_width"]
            h_new, h_old = self.diffs["border_height"]
            new_blocks = [[0 for i in range(w_new)] for j in range(h_new)]
            for y in range(min(h_new, h_old)):
                for x in range(min(w_new, w_old)):
                    new_blocks[y][x] = self.border_blocks[y][x]
            self.root.map.footer.borders = new_blocks
            self.root.map.footer.border_width = w_new
            self.root.map.footer.border_height = h_new
        #Tss and tsp
        if "tsp" in self.diffs:
            tsp_new, tsp_old = self.diffs["tsp"]
            if len(tsp_new): self.root.map.footer.tsp_sym = tsp_new
        if "tss" in self.diffs:
            tss_new, tss_old = self.diffs["tss"]
            if len(tss_new): self.root.map.footer.tss_sym = tss_new
        self.root._refresh()
        self.footer_dialog_update()

    def undo(self):
        if "width" in self.diffs and "height" in self.diffs:
            w_new, w_old = self.diffs["width"]
            h_new, h_old = self.diffs["height"]
            self.root.map.footer.blocks = self.map_blocks[:]
            self.root.map.footer.width = w_old
            self.root.map.footer.height = h_old
        if "border_width" in self.diffs and "border_height" in self.diffs:
            w_new, w_old = self.diffs["border_width"]
            h_new, h_old = self.diffs["border_height"]
            self.root.map.footer.borders = self.border_blocks[:]
            self.root.map.footer.border_width = w_old
            self.root.map.footer.border_height = h_old
        #Tss and tsp
        if "tsp" in self.diffs:
            tsp_new, tsp_old = self.diffs["tsp"]
            if len(tsp_old): self.root.map.footer.tsp_sym = tsp_old
        if "tss" in self.diffs:
            tss_new, tss_old = self.diffs["tss"]
            if len(tss_old): self.root.map.footer.tss_sym = tss_old
        self.root._refresh()
        self.footer_dialog_update()


class Action_event_add(Action_event):
    """ Action class for creation of event """
    def __init__(self, root: Pymap_gui, stype):
        super().__init__(root, stype, None)
        self.id = self._get_element_count(stype)

    def do(self):
        if self.stype == "Person": self.root.map.persons.append(mapevent.Map_event_person())
        elif self.stype == "Warp": self.root.map.warps.append(mapevent.Map_event_warp())
        elif self.stype == "Trigger": self.root.map.triggers.append(mapevent.Map_event_trigger())
        elif self.stype in ("Sign", "SignScript", "SignItem"): self.root.map.signposts.append(mapevent.Map_event_sign())
        self.root._select_event(self.stype, self.id)
        self.root._refresh_event_map()

    def undo(self):
        if self.stype == "Person": self.root.map.persons.remove(self._get_event())
        elif self.stype == "Warp": self.root.map.warps.remove(self._get_event())
        elif self.stype == "Trigger": self.root.map.triggers.remove(self._get_event())
        elif self.stype in ("Sign", "SignScript", "SignItem"): self.root.map.signposts.remove(self._get_event())
        id = self.id - 1
        if id < 0:
            self.root._select_event(self.stype, 0)
        else:
            self.root._select_event(self.stype, id)
        self.root._refresh_event_map()
        

class Action_event_remove(Action_event):
    """ Action class for removal of event """
    def __init__(self, root: Pymap_gui, stype, id):
        super().__init__(root, stype, id)
        self.event = self._get_event()

    def do(self):
        id = self.id
        if self.stype == "Person":
            self.root.map.persons.remove(self.event)
            if id >= len(self.root.map.persons): id = len(self.root.map.persons) - 1
        elif self.stype == "Warp":
            self.root.map.warps.remove(self.event)
            if id >= len(self.root.map.warps): id = len(self.root.map.warps) - 1
        elif self.stype == "Trigger":
            self.root.map.triggers.remove(self.event)
            if id >= len(self.root.map.triggers): id = len(sel.root.map.triggers) - 1
        elif self.stype in ("Sign", "SignScript", "SignItem"):
            self.root.map.signposts.remove(self.event)
            if id >= len(self.root.map.signposts): id = len(self.root.map.signposts) - 1
        if id < 0:
            self.root._select_event(self.stype, 0)
        else:
            self.root._select_event(self.stype, id)
        self.root._refresh_event_map()

    def undo(self):
        if self.stype == "Person":
            self.root.map.persons.insert(self.id, self.event)
        elif self.stype == "Warp":
            self.root.map.persons.insert(self.id, self.event)
        elif self.stype == "Trigger":
            self.root.map.persons.insert(self.id, self.event)
        elif self.stype in ("Sign", "SignScript", "SignItem"):
            sself.root.map.persons.insert(self.id, self.event)
        self.root._select_event(self.stype, self.id)
        self.root._refresh_event_map()

class Action_event_change(Action_event):
    """ Action class for event change """
    def __init__(self, root: Pymap_gui, stype, id, diffs):
        super().__init__(root, stype, id)
        self.root = root
        self.stype = stype
        self.id = id
        self.diffs = diffs


    def do(self):
        event = self._get_event()
        for attr in self.diffs:
            new, old = self.diffs[attr]
            event.__setattr__(attr, new)
        self._update()
    
    def undo(self):
        event = self._get_event()
        for attr in self.diffs:
            new, old = self.diffs[attr]
            event.__setattr__(attr, old)
        self._update()

class Action_border_change(Action):
    """ Action class for border change """
    def __init__(self, root: Pymap_gui, x, y, blocks):
        super().__init__()
        self.root = root
        self.x0 = x
        self.y0 = y
        self.blocks = blocks
        dx, dy = root.block_selection_size
        self.block_selection_size = dx, dy
        self.last_blocks = _arrrange(root.map.footer.borders, self.x0, self.y0, self.x0 + dx, self.y0 + dy, filter=(lambda v: v & 0x3FF))

    def do(self):
        _arrpaste(self.root.map.footer.borders, self.blocks, self.x0, self.y0, filter=(lambda c, t: (c &~0x3FF) | (t & 0x3FF)))
        self.root._refresh_borders()

    def undo(self):
        _arrpaste(self.root.map.footer.borders, self.last_blocks, self.x0, self.y0, filter=(lambda c, t: (c &~0x3FF) | (t & 0x3FF)))
        self.root._refresh_borders()
        
class Action_map_change(Action):
    """ Action class for map change """
    def __init__(self, root: Pymap_gui, x, y, blocks):
        super().__init__()
        self.root = root
        self.x0 = x
        self.y0 = y
        self.blocks = blocks
        dx, dy = root.block_selection_size
        self.block_selection_size = dx, dy
        self.last_blocks = _arrrange(root.map.footer.blocks, self.x0, self.y0, self.x0 + dx, self.y0 + dy, filter=(lambda v: v & 0x3FF))

    def do(self):
        _arrpaste(self.root.map.footer.blocks, self.blocks, self.x0, self.y0, filter=(lambda c, t: (c &~0x3FF) | (t & 0x3FF)))
        dx, dy = self.block_selection_size
        if self.root.context == MAP and dx == 1 and dy == 1: self.root._refresh_map_block(self.x0, self.y0)
        else: self.root._refresh_context_map()

    def undo(self):
        _arrpaste(self.root.map.footer.blocks, self.last_blocks, self.x0, self.y0, filter=(lambda c, t: (c &~0x3FF) | (t & 0x3FF)))
        dx, dy = self.block_selection_size
        if self.root.context == MAP and dx == 1 and dy == 1: self.root._refresh_map_block(self.x0, self.y0)
        else: self.root._refresh_context_map()

        

class Action_map_fill(Action):
    """ Action class for map fill """
    def __init__(self, root: Pymap_gui, x, y, block):
        super().__init__()
        self.root = root
        self.x = x
        self.y = y
        self.block = block & 0x3FF
        self.last_block = self.root.map.footer.blocks[y][x] & 0x3FF
        self.coordinates = self._find_area(x, y)

    def _find_area(self, x, y):
        """ Finds the area connected to the start block (x, y) that have the same block_id"""
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        block = self.root.map.footer.blocks[y][x] & 0x3FF
        visited = set()
        stack = [(x, y)]
        while len(stack):
            x, y = stack.pop()
            if (x, y) in visited: continue #Never explore one coordinate twice
            for i, j in neighbours:
                x1, y1 = x + j, y + i
                if x1 in range(self.root.map.footer.width) and y1 in range(self.root.map.footer.height) and (self.root.map.footer.blocks[y1][x1] & 0x3FF) == block:
                    #Push this coordinate
                    stack.append((x1, y1))
            visited.add((x, y))
        return visited  

    def do(self):
        for x, y in self.coordinates: 
            self.root.map.footer.blocks[y][x] = (self.root.map.footer.blocks[y][x] & ~0x3FF) | (self.block & 0x3FF)
        self.root._refresh_context_map()

    def undo(self):
        for x, y in self.coordinates:
            self.root.map.footer.blocks[y][x] = (self.root.map.footer.blocks[y][x] & ~0x3FF) | (self.last_block & 0x3FF)
        self.root._refresh_context_map()


class Action_level_change(Action):
    """ Action level change """
    def __init__(self, root: Pymap_gui, x, y, level):
        super().__init__()
        self.root = root
        self.x = x
        self.y = y
        self.level = level
        self.level_old = self.root.map.footer.blocks[y][x] >> 10

    def do(self):
        self.root.map.footer.blocks[self.y][self.x] &= ~0xFC00
        self.root.map.footer.blocks[self.y][self.x] |= self.level << 10
        self.root._refresh_level_map_coordinate(self.x, self.y)
    
    def undo(self):
        self.root.map.footer.blocks[self.y][self.x] &= ~0xFC00
        self.root.map.footer.blocks[self.y][self.x] |= self.level_old << 10
        self.root._refresh_level_map_coordinate(self.x, self.y)

class Action_level_fill(Action):
    """ Action class for map fill (level map)"""
    def __init__(self, root: Pymap_gui, x, y, level):
        super().__init__()
        self.root = root
        self.x = x
        self.y = y
        self.level = level & 0x3F
        self.last_level = self.root.map.footer.blocks[y][x] >> 10
        self.coordinates = self._find_area(x, y)

    def _find_area(self, x, y):
        """ Finds the area connected to the start block (x, y) that have the same level"""
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        level = self.root.map.footer.blocks[y][x] >> 10
        visited = set()
        stack = [(x, y)]
        while len(stack):
            x, y = stack.pop()
            if (x, y) in visited: continue #Never explore one coordinate twice
            for i, j in neighbours:
                x1, y1 = x + j, y + i
                if x1 in range(self.root.map.footer.width) and y1 in range(self.root.map.footer.height) and (self.root.map.footer.blocks[y1][x1] >> 10) == level:
                    #Push this coordinate
                    stack.append((x1, y1))
            visited.add((x, y))
        return visited  

    def do(self):
        for x, y in self.coordinates:
            self.root.map.footer.blocks[y][x] = (self.root.map.footer.blocks[y][x] & ~0xFC00) | (self.level << 10)
        self.root._refresh_level_map()

    def undo(self):
        for x, y in self.coordinates:
            self.root.map.footer.blocks[y][x] = (self.root.map.footer.blocks[y][x] & ~0xFC00) | (self.last_level << 10)
        self.root._refresh_level_map()
      

def _arrpaste(dst, src, x0, y0, filter=(lambda current, top: top)):
    """ Pastes src into dst at coordinates x0, y0 (where dst and src are defined), executes a callback for each defined coordinate if present
    The filter function is evaluated for the old element, the element to paste and returns the element that is put in the array"""
    for y in range(y0, y0 + len(src)):
        if y >= len(dst): break
        for x in range(x0, x0 + len(src[y - y0])):
            if x >= len(dst[y]): break
            dst[y][x] = filter(dst[y][x], src[y - y0][x - x0])

def _arrrange(src, x0, y0, x1, y1, filter=(lambda v: v)):
    """ Extracts an array range from source"""
    dst = [[-1 for x in range(x0, x1)] for y in range(y0, y1)]
    for y in range(y0, y1):
        if y >= len(src): break
        for x in range(x0, x1):
            if x >= len(src[y]): break
            dst[y - y0][x - x0] = filter(src[y][x])
    return dst
    
def _arriter(src, f):
    """ Applies a function f to all elements of the array """
    for e in src: f(e)

def _strcap(s):
    """ Capitalizes every char in a string """
    return "".join([c.capitalize() for c in s])

def shell(args, ow_imgs):
    """ Shell call for the gui """
    global BLOCK_CANVAS_SIZE
    global MAP_CANVAS_SIZE
    try:
        opts, args = getopt.getopt(args, "h", ["help"])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "help"):
            print("Usage python pymap.py [projectfile=STDPROJ]")
            return
    
    root = tkinter.Tk()
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    block_canvas_width = 8 * 16 #Always 8 Tiles
    block_canvas_height = int(screen_height * 0.7)
    map_canvas_width = int((screen_width - block_canvas_width) * 0.73)
    map_canvas_height = int(screen_height * 0.85)
    BLOCK_CANVAS_SIZE = block_canvas_width, block_canvas_height
    MAP_CANVAS_SIZE = map_canvas_width, map_canvas_height
    root.wm_title("pymap")
    if len(args):
        proj = args[0]
        proj = project.Project.load_project(proj)
    else:
        proj = project.Project.load_project(STDPROJ)
    gui = Pymap_gui(root, ow_imgs, proj)
    
    root.mainloop()
    









