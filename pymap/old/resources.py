import tkinter.ttk as ttk
import tkinter
from . import project
from . import tileset, image, mapheader
from . import tkinterx
import os

UNDEFINED_NAMESPACE = -1

class ResourcesBrowser(tkinter.Frame):

    """ Creates a frame for browsing resources. All open maps and tilesets should 
    be saved before editing those resource (to keep sync between the resource 
    browser and the file that are curently edited)."""
    def __init__(self, root, proj: project.Project, gui, **kwargs):
        super().__init__(root, **kwargs)
        self.project = proj
        self.root = root
        self.tree = None
        self.gui = gui
        self.refresh()


    def refresh(self):
        """ Refreshes the resource tree """
        if self.tree:
            self.tree.grid_remove()
        self.tree = ttk.Treeview(self.root, selectmode='browse')
        self.tree_items = {} #Maps tree node ids to item classes (to actually be able to hold information)
        self.tree.heading("#0", text="Resources", anchor=tkinter.W)
        #self.tree.heading("#1", text="Filepath", anchor=tkinter.W)
        
        self.popup_menu = tkinter.Menu(self, tearoff=0)

        #Map view
        map_root = self.tree.insert('', 'end', text="Banks")
        self.tree_items[map_root] = MapRootItem(map_root)

        ns_root = self.tree.insert('', 'end', text="Namespaces")
        self.tree_items[ns_root] = NsRoot()
        self.ns_roots = {} #Links to iids for every namespace root
        map_namespaces = self.project.constants.values("map_namespaces")
        for map_namespace in map_namespaces:
            self.ns_roots[map_namespace] = self.tree.insert(ns_root, 'end', text=map_namespace)
            self.tree_items[self.ns_roots[map_namespace]] = NsRoot()
        self.ns_roots[UNDEFINED_NAMESPACE] = self.tree.insert(ns_root, 'end', text="undefined namespace")

        for bank in sorted(self.project.banks.keys()):
            bank_node = self.tree.insert(map_root, 'end', text=str(bank))
            self.tree_items[bank_node] = BankItem(bank, bank_node)
            for mapid in sorted(self.project.banks[bank].keys()):
                symbol, path, namespace, footer_id = self.project.banks[bank][mapid]
                map_node = self.tree.insert(bank_node, 'end', text=str(mapid) + " : " + symbol)
                if namespace in self.ns_roots:
                    ns_map_node = self.tree.insert(self.ns_roots[namespace], 'end', text=str(bank) + "." + str(mapid) + " : " + symbol)
                else:
                    ns_map_node = self.tree.insert(self.ns_roots[UNDEFINED_NAMESPACE], 'end', text=str(bank) + "." + str(mapid) + " : " + symbol)
                self.tree_items[map_node] = MapItem(bank, mapid, path, map_node, ns_map_node, self)
                self.tree_items[ns_map_node] = self.tree_items[map_node]
                


        #Tileset view
        ts_root = self.tree.insert('', 'end', text="Tilesets")
        self.tree_items[ts_root] = TilesetRootItem(ts_root)
        for ts in sorted(self.project.tilesets.keys()):
            path = self.project.tilesets[ts]
            ts_node = self.tree.insert(ts_root, 'end', text=ts)
            self.tree_items[ts_node] = TilesetItem(ts, path, ts_node)
        
        #Gfx view
        gfx_root = self.tree.insert('', 'end', text="Gfx")
        self.tree_items[gfx_root] = GfxRootItem(gfx_root)
        for gfx in sorted(self.project.images.keys()):
            path = self.project.images[gfx]
            gfx_node = self.tree.insert(gfx_root, 'end', text=gfx)
            self.tree_items[gfx_node] = GfxItem(gfx, path, gfx_node)

        def popup(event):
            iid = self.tree.identify_row(event.y)
            if iid:
                self.tree.selection_set(iid)
                item = self.tree_items[iid]
                item.popup(self, event)
        
        def double_click(event):
            iid = self.tree.identify_row(event.y)
            if iid:
                item = self.tree_items[iid]
                if isinstance(item, MapItem):
                    self.gui.open_map(item.bank, item.mapid)


        self.tree.bind("<Button-3>", popup)
        self.tree.bind("<Double-1>", double_click)

        self.tree.grid(row=0, column=0, sticky=tkinter.NSEW)
            


class NsRoot:
    """ Stub for the namespace root"""
    def __init__(self): pass
    def popup(self, frame: ResourcesBrowser, e): pass

class MapItem:
    
    def __init__(self, bank, mapid, path, iid, ns_iid, frame: ResourcesBrowser):
        self.bank = bank
        self.mapid = mapid
        self.path = path
        self.iid = iid
        self.ns_iid = ns_iid
        self.frame = frame

    def remove(self):
        """ Removes the map from the project and the tree """
        if tkinter.messagebox.askyesno(title="Remove map", message="Do you really want to remove this map from the project?"):
            if self.frame.gui.map and self.frame.gui.bank == self.bank and self.frame.gui.mapid == self.mapid and self.frame.gui.map_has_changed():
                if tkinter.messagebox.askyesno(title="Save changes before removal", message="Deleting the current map will result in a loss of all changes. Do you want to save the map before removing it?"):
                    self.frame.gui.file_save()
                self.frame.gui.map = None
                self.frame.gui._refresh()
                    
            _, path, _, _, =  self.frame.project.banks[self.bank].pop(self.mapid)
            self.frame.tree.delete(self.iid)
            self.frame.tree.delete(self.ns_iid)
            self.frame.project.save_project()
            if tkinter.messagebox.askyesno(title="Delete map on disk", message="Do you want to delete the map on the disk (it can not be restored)?", default="no"):
                os.remove(path)
            

    def popup(self, frame: ResourcesBrowser, e):
        
        def change_namespace():
            """ Changes the namespace of the map """
            mh = mapheader.load(frame.project.get_map_path(self.bank, self.mapid), frame.project, instanciate_ts=False)
            _, path, _, _ = frame.project.banks[self.bank][self.mapid]
            map_namespaces = frame.project.constants.values("map_namespaces")
            new_namespace = tkinterx.askcombobox(frame.gui.root, "Change namespace", list(map_namespaces), default=mh.name_bank)
            if mh.name_bank == new_namespace or new_namespace == None: return #No changes whatsoever
            
            new_root = new_namespace
            if new_namespace not in map_namespaces:
                new_root = UNDEFINED_NAMESPACE
            
            mh.name_bank = new_namespace
            mh.save(path)

            if frame.gui.bank == self.bank and frame.gui.mapid == self.mapid and frame.gui.map:
                frame.gui.map.name_bank = new_namespace

            frame.tree.delete(self.ns_iid)
            self.ns_iid = frame.tree.insert(frame.ns_roots[new_root], 'end', text=str(self.bank) + "." + str(self.mapid) + " : " + mh.symbol)
            frame.tree_items[self.ns_iid] = self

        def refractor():
            """ Refractors the symbol of the map """
            symbol = None
            while not symbol:
                symbol = tkinter.simpledialog.askstring("Refractor map symbol", "Refine the unique symbol for the map")
                if symbol == None: 
                    return
                symbol_location = frame.project.get_map_location_by_symbol(symbol)
                if symbol_location:
                    tkinter.messagebox.showerror(title="Symbol already definied", message="The symbol '" + symbol + "' is already definied at " + str(symbol_location))
                    symbol = None
                elif not len(symbol):
                    tkinter.messagebox.showerror(title="Empty symbol", message="You must not define an empty symbol")
                    symbol = None
            
            if frame.gui.map:
                if frame.gui.bank == self.bank and frame.gui.mapid == self.mapid:
                    if not tkinter.messagebox.askyesno(title="Project needs to be saved", message="To refractor the symbol of the current map this map will be saved after refractoring. Continue?"): return
                    frame.gui.map.symbol = symbol
                    frame.gui.file_save()

            _, path, namespace, footer_id = frame.project.banks[self.bank][self.mapid]
            frame.project.banks[self.bank][self.mapid] = symbol, path, namespace, footer_id
            frame.project.save_project()
            frame.tree.item(self.iid, text=symbol)

        

        popup_menu = tkinter.Menu(frame, tearoff=0)
        popup_menu.add_command(label="Remove", command=self.remove)
        popup_menu.add_command(label="Refractor", command=refractor)
        popup_menu.add_command(label="Change namespace", command=change_namespace)
        popup_menu.tk_popup(e.x_root, e.y_root)


class BankItem:
    
    def __init__(self, bank, iid): 
        self.bank = bank
        self.iid = iid
    def popup(self, frame: ResourcesBrowser, e):
        
        def remove():
            """ Deletes the bank (and thus all references to maps)  but not anything on the disk (simply to prevent mass deletion)"""
            if tkinter.messagebox.askyesno(title="Remove bank", message="Do you really want to remove this bank? It will result in a loss of all references."):
                children = frame.tree.get_children(self.iid)
                if len(children): frame.tree.delete(children)
                frame.tree.delete(self.iid)
                #Delete all children in their respective namespaces
                for child in children:
                    frame.tree.delete(frame.tree_items[child].ns_iid)

                delete_on_disk = tkinter.messagebox.askyesno(title="Delete bank on disk", message="Do you want to delete all maps on your disk (they can not be restored)?", default="no")
                #Check if the gui holds one of the deleted maps and if deassing the map
                if frame.gui.map:
                    for child in children:
                        bank, mapid = frame.tree_items[child].bank, frame.tree_items[child].mapid
                        symbol, path, _, _ = frame.project.banks[bank][mapid]
                        if frame.gui.bank == bank and frame.gui.mapid == mapid:
                            frame.gui.map = None
                            frame.gui._refresh()
                        if delete_on_disk: os.remove(path)
                frame.project.banks.pop(self.bank)
                frame.project.save_project()

        def _new():
            """ Creates a new empty map """
            #Enter a map id
            mapid = tkinter.simpledialog.askinteger("Enter map id", "Enter map id of map to create")
            if mapid in frame.project.banks[self.bank]:
                tkinter.messagebox.showerror(title="Invalid map id", message="There already is map with this id")
                return
            if mapid not in range(0, 256):
                tkinter.messagebox.showerror(title="Invalid map id", message="The map id must be integer value in range 0 to 255 (included)!")
                return
            #Enter a symbol
            symbol = None
            while not symbol:
                symbol = tkinter.simpledialog.askstring("Define map symbol", "Define the unique symbol for the map to create")
                if symbol == None: 
                    return
                symbol_location = frame.project.get_map_location_by_symbol(symbol)
                if symbol_location:
                    tkinter.messagebox.showerror(title="Symbol already definied", message="The symbol '" + symbol + "' is already definied at " + str(symbol_location))
                    symbol = None
                elif not len(symbol):
                    tkinter.messagebox.showerror(title="Empty symbol", message="You must not define an empty symbol")
                    symbol = None
            opts = {
                "initialdir" : os.getcwd(),
                "filetypes" : [('pymap mapheader files', '.pmh')],
                "title" : "Select map to create",
                "parent" : frame
            }
            path = tkinter.filedialog.asksaveasfilename(**opts)
            if path and len(path):
                path = project.Project._sanitized_relative_path(path)
                mh = mapheader.Mapheader()
                mh.symbol = symbol
                mh.id = frame.project.get_smallest_availible_foooter_id()
                frame.project.save_map(self.bank, mapid, mh, path)
                frame.project.save_project()
                #Add a tree item
                ciid = frame.tree.insert(self.iid, 'end', text=str(mapid) + " : " + symbol)
                ns_ciid = frame.tree.insert(frame.ns_roots[UNDEFINED_NAMESPACE], 'end', text=str(self.bank) + "." + str(mapid) + " : " + symbol)
                frame.tree_items[ciid] = MapItem(self.bank, mapid, path, ciid, ns_ciid, frame)
                frame.tree_items[ns_ciid] = frame.tree_items[ciid]



        def _import():
            """ Imports a map into this bank (pmh file)"""
            mapid = tkinter.simpledialog.askinteger("Enter map id", "Enter map id of map to import")
            if mapid in frame.project.banks[self.bank]:
                tkinter.messagebox.showerror(title="Invalid map id", message="There already is map with this id")
                return
            if mapid != None:
                opts = {
                    "initialdir" : os.getcwd(),
                    "filetypes" : [('pymap mapheader files', '.pmh')],
                    "title" : "Select map to import",
                    "parent" : frame
                }
                path = tkinter.filedialog.askopenfilename(**opts)
                if path and len(path):
                    path = project.Project._sanitized_relative_path(path)
                    mh = mapheader.load(path, frame.project)
                    symbol_location = frame.project.get_map_location_by_symbol(mh.symbol)
                    if symbol_location:
                        tkinter.messagebox.showerror(title="Symbol already defined", message="There is already a map with global symbol '" + mh.symbol + "' at " + str(symbol_location))
                        return
                    frame.project.save_map(self.bank, mapid, mh, path)
                    frame.project.save_project()
                    #Find the map namespace
                    namespace = mh.name_bank if mh.name_bank in frame.ns_roots else UNDEFINED_NAMESPACE
                    #Add an tree item
                    ciid = frame.tree.insert(self.iid, 'end', text=str(mapid) + " : " + mh.symbol)
                    ns_ciid = frame.tree.insert(frame.ns_roots[namespace], 'end', text=str(self.bank) + "." + str(mapid) + " : " + mh.symbol)
                    frame.tree_items[ciid] = MapItem(self.bank, mapid, path, ciid, ns_ciid, frame)
                    frame.tree_items[ns_ciid] = frame.tree_items[ciid]

        popup_menu = tkinter.Menu(frame, tearoff=0)
        popup_menu.add_command(label="New", command=_new)
        popup_menu.add_command(label="Import", command=_import)
        popup_menu.add_command(label="Remove", command=remove)
        popup_menu.tk_popup(e.x_root, e.y_root)
        

class MapRootItem:
    
    def __init__(self, iid):
        self.iid = iid
    def popup(self, frame: ResourcesBrowser, e):
        
        def new_bank():
            """ Creates a new empty bank """
            bank = None
            while bank == None:
                bank = tkinter.simpledialog.askinteger("Create new bank", "Enter bank number you want to create")
                if bank == None: return
                if bank not in range(0, 256):
                    tkinter.messagebox.showerror(title="Invalid bank number", message="You entered an invalid bank number")
                    bank = None
                if bank in frame.project.banks:
                    tkinter.messagebox.showerror(title="Bank already used", message="You can not create bank with this number since it is already used")
                    bank = None
            print(bank)
            frame.project.banks[bank] = {}
            frame.project.save_project()
            ciid = frame.tree.insert(self.iid, 'end', text=str(bank))
            frame.tree_items[ciid] = BankItem(bank, ciid)

        popup_menu = tkinter.Menu(frame, tearoff=0)
        popup_menu.add_command(label="New bank", command=(lambda: new_bank()))
        popup_menu.tk_popup(e.x_root, e.y_root)

class TilesetRootItem:
    
    def __init__(self, iid):
        self.iid = iid 

    def popup(self, frame: ResourcesBrowser, e): 

        def _new():
            """ Creates a new tileset """
            opts = {
                "initialdir" : os.getcwd(),
                "filetypes" : [('pymap tileset files', '.pts')],
                "title" : "Save new tileset file",
                "parent" : frame
            }
            path = tkinter.filedialog.asksaveasfilename(**opts)
            if path and len(path):
                path = project.Project._sanitized_relative_path(path)
                primary = tkinter.messagebox.askyesno(title="Tileset type", message="Create a primary tileset (tsp)?")
                symbol = None
                while not symbol:
                    symbol = tkinter.simpledialog.askstring("Tileset symbol", "Define a unique symbol for the tileset")
                    if symbol == None: 
                        return
                    if symbol in frame.project.tilesets:
                        tkinter.messagebox.showerror(title="Symbol already definied", message="The symbol '" + symbol + "' is already definied!")
                        symbol = None
                    elif not len(symbol):
                        tkinter.messagebox.showerror(title="Empty symbol", message="You must not define an empty symbol")
                        symbol = None
                ts = tileset.Tileset(primary)
                ts.gfx = "0"
                ts.symbol = symbol
                frame.project.save_tileset(symbol, ts, path)
                frame.project.save_project()
                child_iid = frame.tree.insert(self.iid, 'end', text=symbol)
                frame.tree_items[child_iid] = TilesetItem(symbol, path, child_iid)
                frame.gui._refresh()

        def _import():
            """ Imports a present tileset from the disk """
            opts = {
                "initialdir" : os.getcwd(),
                "filetypes" : [('pymap tileset files', '.pts')],
                "title" : "Save new tileset file",
                "parent" : frame
            }
            path = tkinter.filedialog.askopenfilename(**opts)
            if path and len(path):
                path = project.Project._sanitized_relative_path(path)
                ts = tileset.from_file(path)
                symbol = ts.symbol
                if symbol in frame.project.tilesets:
                    tkinter.messagebox.showerror(title="Symbol already defined", message="The symbol of the tileset to import '" + symbol + "' is already defined! Change the existing symbol first!")
                    #We do not allow to change the symbol during import as it may result as associating the same file to two different symbols which would result in a confilict for the compiler
                    #As which symbol should the file be compiled?
                    return
                frame.project.save_tileset(symbol, ts, path)
                frame.project.save_project()
                child_iid = frame.tree.insert(self.iid, 'end', text=symbol)
                frame.tree_items[child_iid] = TilesetItem(symbol, path, child_iid)
                frame.gui._refresh()

        

        popup_menu = tkinter.Menu(frame.root, tearoff=0)
        popup_menu.add_command(label="New", command=_new)
        popup_menu.add_command(label="Import", command=_import)
        popup_menu.tk_popup(e.x_root, e.y_root)

class TilesetItem:
    
    def __init__(self, symbol, path, iid):
        self.iid = iid
        self.symbol = symbol
        self.path = path
    

    def popup(self, frame: ResourcesBrowser, e):
        
        def update():
            """ Updates the item in tree view """
            frame.tree.item(self.iid, text=self.symbol)

        """
        Somewhat useless I guess? (just use remove and reimport)
        def change_path():
             Changes the path of an item in tree view by dialog.
            Instanciates the selected pts file (for sanity checking),
            saves it and if both succeded changes the path attribute 
            opts = {
                "initialdir" : os.path.dirname(self.path),
                "filetypes" : [('pymap tileset files', '.pts')],
                "title" : "Select tileset path",
                "parent" : frame
            }
            path = tkinter.filedialog.askopenfilename(**opts)
            if path and len(path):
                path = project.Project._sanitized_relative_path(path)
                try:
                    t = tileset.from_file(path) #Try to instanciate to only allow valid tilesets
                    frame.project.save_tileset(self.symbol, t, path)
                    frame.project.save_project()
                    self.path = path
                    update()
                    frame.gui._refresh()
                except Exception as ex:
                    tkinter.messagebox.showerror(title="Unable to open tileset file", message="Unable to open tileset file '" + path + "'\n " + str(ex))"""
            
        def refractor():
            """ Refractors the symbol of the tileset """
            symbol = None
            while not symbol:
                symbol = tkinter.simpledialog.askstring("Refractor tileset symbol", "Refine the unique symbol for the tileset")
                if symbol == None: 
                    return
                if symbol in frame.project.tilesets:
                    tkinter.messagebox.showerror(title="Symbol already definied", message="The symbol '" + symbol + "' is already definied!")
                    symbol = None
                elif not len(symbol):
                    tkinter.messagebox.showerror(title="Empty symbol", message="You must not define an empty symbol")
                    symbol = None
            frame.project.refractor_tileset_symbol(symbol, self.symbol)
            self.symbol = symbol
            update()

        def remove():
            """ Removes a tileset """
            if tkinter.messagebox.askyesno(title="Aprove deletion of tileset", message="Do you really want to remove the tileset from this project?"):
                frame.project.remove_tileset(self.symbol)
                frame.project.save_project()
                frame.tree.delete(self.iid)
                frame.gui._refresh()
                if tkinter.messagebox.askyesno(title="Delete the pts file", message="Do you want to remove the pts file permanently from the disk?", default=tkinter.NO):
                    if os.path.isfile(self.path): os.remove(self.path)
                    else: tkinter.messagebox.showerror(title="File does not exist", message="The linked tileset file '" + self.path + "' does not exist and could not be deleted")
                
        popup_menu = tkinter.Menu(frame.root, tearoff=0)
        #popup_menu.add_command(label="Change path", command=change_path)
        popup_menu.add_command(label="Remove", command=remove)
        popup_menu.add_command(label="Refractor", command=refractor)
        popup_menu.tk_popup(e.x_root, e.y_root)

class GfxRootItem:
    
    def __init__(self, iid):
        self.iid = iid 

    def popup(self, frame: ResourcesBrowser, e): 
        
        def _import():
            """ Imports a present image from the disk """
            opts = {
                "initialdir" : os.getcwd(),
                "filetypes" : [('portable network graphics', '.png')],
                "title" : "Import gfx",
                "parent" : frame
            }
            path = tkinter.filedialog.askopenfilename(**opts)
            if path and len(path):
                path = project.Project._sanitized_relative_path(path)
                img = image.Image()
                try: img.load_image_file(path)
                except Exception as e:
                    tkinter.messagebox.showerror(title="Unable to open gfx", message="Unable to open gfx file '" + path + "': " + str(e))
                    return
                symbol = None
                while not symbol:
                    symbol = tkinter.simpledialog.askstring("Gfx symbol", "Define a unique symbol for the gfx")
                    if symbol == None: 
                        return
                    if symbol in frame.project.images:
                        tkinter.messagebox.showerror(title="Symbol already definied", message="The symbol '" + symbol + "' is already definied!")
                        symbol = None
                    elif not len(symbol):
                        tkinter.messagebox.showerror(title="Empty symbol", message="You must not define an empty symbol")
                        symbol = None
                frame.project.save_image(symbol, path)
                frame.project.save_project()
                child_iid = frame.tree.insert(self.iid, 'end', text=symbol)
                frame.tree_items[child_iid] = GfxItem(symbol, path, child_iid)
                frame.gui._refresh()

            
        popup_menu = tkinter.Menu(frame.root, tearoff=0)
        popup_menu.add_command(label="Import", command=_import)
        popup_menu.tk_popup(e.x_root, e.y_root)

class GfxItem:
    
    def __init__(self, symbol, path, iid):
        self.iid = iid
        self.symbol = symbol
        self.path = path

    def popup(self, frame: ResourcesBrowser, e):
        
        def update():
            """ Updates the item in tree view """
            frame.tree.item(self.iid, text=self.symbol)
            frame.gui._refresh()

        """
        Somewhat useless I guess? (just use remove and repimport)
        def change_path():
             Changes the path of an item in tree view by dialog.
            Instanciates the selected .png file (for sanity checking),
            saves it and if both succeded changes the path attribute 
            opts = {
                "initialdir" : os.path.dirname(self.path),
                "filetypes" : [('portable network graphics', '.png')],
                "title" : "Select gfx path",
                "parent" : frame
            }
            path = tkinter.filedialog.askopenfilename(**opts)
            if path and len(path):
                path = project.Project._sanitized_relative_path(path)
                try:
                    img = image.Image()
                    img.load_image_file(path) #Try to instanciate to only allow valid tilesets
                    frame.project.save_image(self.symbol, path)
                    frame.project.save_project()
                    self.path = path
                    update()
                except Exception as ex:
                    tkinter.messagebox.showerror(title="Unable to open image file", message="Unable to open image file '" + path + "'\n " + str(ex))"""
        
        def remove():
            """ Removes an image """
            if tkinter.messagebox.askyesno(title="Aprove deletion of gfx", message="Do you really want to remove the gfx from this project?"):
                frame.project.remove_image(self.symbol)
                frame.project.save_project()
                frame.tree.delete(self.iid)
                if tkinter.messagebox.askyesno(title="Delete the gfx file", message="Do you want to remove the gfx file permanently from the disk?"):
                    if os.path.isfile(self.path): os.remove(self.path)
                    else: tkinter.messagebox.showerror(title="File does not exist", message="The linked gfx file '" + self.path + "' does not exist and could not be deleted")
                frame.gui._refresh()
                
        popup_menu = tkinter.Menu(frame.root, tearoff=0)
        #popup_menu.add_command(label="Change path", command=change_path)
        popup_menu.add_command(label="Remove", command=remove)
        popup_menu.tk_popup(e.x_root, e.y_root)