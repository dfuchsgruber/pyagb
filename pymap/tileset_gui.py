#!/usr/bin/python3

import tkinter
import tkinter.simpledialog
from . import tkinterx
from tkinter import messagebox, filedialog
from . import tileset, palette
import os, getopt
import PIL.Image as PImage
import PIL.ImageTk as ImageTk
import numpy as np
import time
from threading import Thread
from . import project, resources
import tkinter.ttk as ttk
import png

CANVAS_ZOOM = 3
CANVAS_WIDTH = 128 * CANVAS_ZOOM
CANVAS_HEIGHT_TSP = 1280 * CANVAS_ZOOM
CANVAS_HEIGHT_TSS = 768 * CANVAS_ZOOM
CANVAS_HEIGHT = 512 * CANVAS_ZOOM


class Tileset_gui(tkinter.Frame):

    def __init__(self, root, proj=None):

        self.project = proj
        self.primary_tileset = None
        self.secondary_tileset = None
        self.root = root
        self.selected_tiles = [[(0, False, False, 0)]] #Elements are 4-tuples tile, hflip, vflip, pal_id
        self.selected_block_tsp = 0
        self.selected_block_tss = 0
        self.block_to_build_tsp = [(0, False, False, 0)] * 8
        self.block_to_build_tss = [(0, False, False, 0)] * 8
        self.block_to_build_tsp_selected_area = 0, 0, 1, 1
        self.block_to_build_tss_selected_area = 0, 0, 1, 1
        self.tileset_selected_area = 0, 0, 1, 1
        self.selected_block_tsp = 0
        self.selected_block_tss = 0

        self._init_gui(root)


    def _init_gui(self, root):
        tkinter.Frame.__init__(self, root) #Superclass init
        
        self.refresh_title()

        root.minsize(width=460 * CANVAS_ZOOM, height=800)
        root.maxsize(width=460 * CANVAS_ZOOM, height=800)

        #Setup the menu
        menu = tkinter.Menu(root)
        root.config(menu=menu)
        tsp_file_menu = tkinter.Menu(menu, tearoff=0)
        menu.add_cascade(label="Primary", menu=tsp_file_menu)
        tsp_file_menu.add_command(label="Save", command=(lambda : self.file_save(True)))   
        tss_file_menu = tkinter.Menu(menu, tearoff=0)
        menu.add_cascade(label="Secondary", menu=tss_file_menu)
        tss_file_menu.add_command(label="Save", command=(lambda : self.file_save(False)))

        def blocks_extend():
            #Extend the blocks on tss
            bs = tkinter.simpledialog.askinteger("Change #blocks", "Enter new #blocks (max 384)", minvalue=1, maxvalue=384, initialvalue=len(self.secondary_tileset.blocks))
            if bs <= len(self.secondary_tileset.blocks):
                self.secondary_tileset.blocks = self.secondary_tileset.blocks[:bs]
                self.secondary_tileset.behaviours = self.secondary_tileset.behaviours[:bs]
            else:
                bs_additional = bs - len(self.secondary_tileset.blocks)
                self.secondary_tileset.blocks += [[0 for i in range(8)]] * bs_additional
                self.secondary_tileset.behaviours += [["0x0" for i in range(8)]] * bs_additional
            if self.selected_block_tss >= bs:
                self.selected_block_tss = bs - 1
                self.blocks_tss_raw_pil_image = PImage.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT_TSP), "black")
            self.refresh_block_selection(False)
            self.refresh_block_edit(False)
        tss_file_menu.add_command(label="Change #blocks", command=blocks_extend)

        
        #Setup the labelframes
        #-----------------------------------------------------------------
        self.frame_tsp = ttk.Labelframe(root, text="Primary tileset (tsp)")
        self.frame_tss = ttk.Labelframe(root, text="Secondary tileset (tss)")
        self.frame_pool = ttk.Labelframe(root, text="Tile pool")
        self.frame_tsp.grid(row=0, column=0, sticky=tkinter.NW)
        self.frame_tss.grid(row=0, column=1, sticky=tkinter.NW)
        self.frame_pool.grid(row=0, column=2, sticky=tkinter.NW)

        #Setup the canvas for the tileset pool
        #-------------------------------------------------------------------
        canvas_frame = tkinter.Frame(self.frame_pool)
        canvas_frame.grid(row=0, column=0, sticky=tkinter.NW)
        self.tileset_canvas = tkinter.Canvas(canvas_frame, width=CANVAS_WIDTH, height=512, scrollregion=(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), background="black")
        self.tileset_canvas_bar_v = tkinter.Scrollbar(canvas_frame, orient=tkinter.VERTICAL)
        self.tileset_canvas_bar_v.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.tileset_canvas_bar_v.config(command=self.tileset_canvas.yview)
        self.tileset_raw_pil_image = PImage.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), "black")
        self.tileset_raw_image = ImageTk.PhotoImage(self.tileset_raw_pil_image)
        self.tileset_canvas_image = self.tileset_canvas.create_image(1, 1, image=self.tileset_raw_image, anchor=tkinter.NW)
        self.tileset_canvas.config(yscrollcommand=self.tileset_canvas_bar_v.set) 
        #Setup properties window for tileset pool
        #-------------------------------------------------------------------
        self.frame_pool_properties = tkinter.Frame(self.frame_pool)
        self.frame_pool_properties.grid(row=1, sticky=tkinter.NW)
        tkinter.Label(self.frame_pool_properties, text="Palette").grid(row=0, column=0, sticky=tkinter.NW)
        self.tileset_pal_combobox = tkinterx.Combobox(self.frame_pool_properties, list(map(str, range(13))), state='readonly')
        self.tileset_pal_combobox.grid(row=0, column=1, columnspan=2, sticky=tkinter.NW)
        self.tileset_pal_combobox.add_callback(lambda: self.refresh_tileset_canvas())
        self.tileset_pal_combobox.set("0", untraced=True)
        tkinter.Button(self.frame_pool_properties, text="Import", command=self.open_pal_dialoge).grid(row=0, column=3, sticky=tkinter.NW)
        self.tile_selection_canvas = tkinter.Canvas(self.frame_pool_properties, width=8 * CANVAS_ZOOM * 2, height=8 * CANVAS_ZOOM * 4, background="black")
        self.tile_selection_raw_image = ImageTk.PhotoImage(PImage.new("RGBA", (8 * CANVAS_ZOOM, 8 * CANVAS_ZOOM), "black"))
        self.tile_selection_canvas_image = self.tile_selection_canvas.create_image(1, 1, image=self.tile_selection_raw_image, anchor=tkinter.NW)
        self.tile_selection_canvas.grid(row=1, column=0, rowspan=2, sticky=tkinter.NW)
        self.tile_selection_hflip_box = tkinterx.Checkbutton(self.frame_pool_properties, "HFlip", onvalue=True, offvalue=False)
        def flip_selection(hflip=False, vflip=False):
            self.selected_tiles = [[(t, (not h if hflip else h), (not v if vflip else v), p) for (t, h, v, p) in line] for line in self.selected_tiles]
            self.refresh_tile_selection()
        self.tile_selection_hflip_box.add_callback(lambda: flip_selection(hflip=True))
        self.tile_selection_hflip_box.grid(row=1, column=1, sticky=tkinter.NW)
        self.tile_selection_vflip_box = tkinterx.Checkbutton(self.frame_pool_properties, "VFlip", onvalue=True, offvalue=False)
        self.tile_selection_vflip_box.add_callback(lambda: flip_selection(vflip=True))
        self.tile_selection_vflip_box.grid(row=1, column=2, sticky=tkinter.NW)
        #Data binding for tileset pool
        x0, y0, x1, y1 = self.tileset_selected_area
        self.tileset_canvas_selection_rectangle = self.tileset_canvas.create_rectangle(x0 * 8 * CANVAS_ZOOM, y0 * 8 * CANVAS_ZOOM, x1 * 8 * CANVAS_ZOOM, y1 * 8 * CANVAS_ZOOM, fill="", outline="red")
        #Define inner methods for mouse drag
        def tile_selection_b_press(event):
            slider_top, slider_bottom = self.tileset_canvas_bar_v.get()
            x, y = event.x, event.y + int(CANVAS_HEIGHT * slider_top)
            x0 = int(x / (8 * CANVAS_ZOOM)) 
            y0 = int(y / (8 * CANVAS_ZOOM))
            self.tileset_selected_area = x0, y0, x0 + 1, y0 + 1
            self.refresh_tile_selection_by_tileset_area()
        def tile_selection_b_motion(event):
            slider_top, slider_bottom = self.tileset_canvas_bar_v.get()
            x, y = event.x, event.y + int(CANVAS_HEIGHT * slider_top)
            x0, y0, _, _ = self.tileset_selected_area
            x1 = int(x / (8 * CANVAS_ZOOM)) + 1
            y1 = int(y / (8 * CANVAS_ZOOM)) + 1
            if x1 > 16: x1 = 16
            if y1 > 64: y1 = 64
            dx, dy = x1 - x0, y1 - y0
            if dx <= 0 or dy <= 0: return
            if dx > 2: x1 = x0 + 2
            if dy > 4: y1 = y0 + 4
            self.tileset_selected_area = x0, y0, x1, y1
            self.refresh_tile_selection_by_tileset_area()
        self.tileset_canvas.bind("<ButtonPress-1>", tile_selection_b_press)
        self.tileset_canvas.bind("<ButtonPress-3>", tile_selection_b_press)
        self.tileset_canvas.bind("<B3-Motion>", tile_selection_b_motion)
        self.tileset_canvas.pack(anchor=tkinter.NW)

        #Setup the two canvases for block selection (one for tsp and tss)
        #-------------------------------------------------------------------
        canvas_frame_blocks_tsp, canvas_frame_blocks_tss = tkinter.Frame(self.frame_tsp), tkinter.Frame(self.frame_tss)
        canvas_frame_blocks_tsp.grid(row=0, column=0, columnspan=2, sticky=tkinter.NW)
        canvas_frame_blocks_tss.grid(row=0, column=0, columnspan=2, sticky=tkinter.NW)
        self.blocks_tsp_canvas = tkinter.Canvas(canvas_frame_blocks_tsp, width=CANVAS_WIDTH, height=512, scrollregion=(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT_TSP), background="black")
        self.blocks_tss_canvas = tkinter.Canvas(canvas_frame_blocks_tss, width=CANVAS_WIDTH, height=512, scrollregion=(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT_TSS), background="black")
        self.blocks_tsp_bar_v = tkinter.Scrollbar(canvas_frame_blocks_tsp, orient=tkinter.VERTICAL)
        self.blocks_tss_bar_v = tkinter.Scrollbar(canvas_frame_blocks_tss, orient=tkinter.VERTICAL)
        self.blocks_tsp_bar_v.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.blocks_tss_bar_v.pack(side=tkinter.RIGHT, fill=tkinter.Y)
        self.blocks_tsp_bar_v.config(command=self.blocks_tsp_canvas.yview)
        self.blocks_tss_bar_v.config(command=self.blocks_tss_canvas.yview)
        self.blocks_tsp_raw_pil_image = PImage.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT_TSP), "black")
        self.blocks_tss_raw_pil_image = PImage.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT_TSS), "black")
        self.blocks_tsp_raw_image = ImageTk.PhotoImage(self.blocks_tsp_raw_pil_image)
        self.blocks_tss_raw_image = ImageTk.PhotoImage(self.blocks_tss_raw_pil_image)
        self.blocks_tsp_canvas_image = self.blocks_tsp_canvas.create_image(1, 1, image=self.blocks_tsp_raw_image, anchor=tkinter.NW)
        self.blocks_tss_canvas_image = self.blocks_tss_canvas.create_image(1, 1, image=self.blocks_tss_raw_image, anchor=tkinter.NW)
        self.blocks_tsp_canvas.config(yscrollcommand=self.blocks_tsp_bar_v.set)
        self.blocks_tss_canvas.config(yscrollcommand=self.blocks_tss_bar_v.set)
        x, y = self.selected_block_tsp & 7, self.selected_block_tsp >> 3
        self.blocks_tsp_canvas_rectangle = self.blocks_tsp_canvas.create_rectangle(x * 16 * CANVAS_ZOOM, y * 16 * CANVAS_ZOOM, (x + 1) * 16 * CANVAS_ZOOM, (y + 1) * 16 * CANVAS_ZOOM, fill="", outline="red")
        x, y = self.selected_block_tss & 7, self.selected_block_tss >> 3
        self.blocks_tss_canvas_rectangle = self.blocks_tss_canvas.create_rectangle(x * 16 * CANVAS_ZOOM, y * 16 * CANVAS_ZOOM, (x + 1) * 16 * CANVAS_ZOOM, (y + 1) * 16 * CANVAS_ZOOM, fill="", outline="red")
        def block_selection_b_press(event, primary):
            t = self.primary_tileset if primary else self.secondary_tileset
            if not t: return
            slider_top, slider_bottom = self.blocks_tsp_bar_v.get() if primary else self.blocks_tss_bar_v.get()
            ex, ey = event.x, event.y + int((CANVAS_HEIGHT_TSP if primary else CANVAS_HEIGHT_TSS) * slider_top)
            x, y = int(ex / (16 * CANVAS_ZOOM)), int(ey / (16 * CANVAS_ZOOM))
            block = 8 * y + x
            if block >= len(t.blocks): return
            if primary: self.selected_block_tsp = block
            else: self.selected_block_tss = block
            self.refresh_block_selection_rectangle(primary)
            #Change the label
            if primary: 
                self.lb_head_block_edit_tsp["text"] = "Block #" + hex(block)
                self.block_to_build_tsp = [(b & 0x3FF, b & 0x800 != 0, b & 0x400 != 0, (b >> 12) & 0xF) for b in t.blocks[block]]
                self.refresh_block_edit(True)
            else: 
                self.lb_head_block_edit_tss["text"] = "Block #" + hex(block + 0x280)
                self.block_to_build_tss = [(b & 0x3FF, b & 0x800 != 0, b & 0x400 != 0, (b >> 12) & 0xF) for b in t.blocks[block]]
                self.refresh_block_edit(False)
        #Define a popup  menu structure
        context_menu_tsp, context_menu_tss = tkinter.Menu(root, tearoff=0), tkinter.Menu(root, tearoff=0)
        
        def create_properties_dialog(primary):
            block_id = self.selected_block_tsp if primary else self.selected_block_tss
            t = self.primary_tileset if primary else self.secondary_tileset
            dialog = tkinter.Toplevel(root)
            dialog.wm_title("Properties of block #" + hex(block_id if primary else block_id + 0x280))
            dialog.minsize(170, 190), dialog.maxsize(170, 190), dialog.resizable(0, 0)
            #dialog.attributes("-toolwindow",1)
            label_texts = ["Behaviour", "Hm usage", "Field_2", "Field_3", "Field_4", "Field_5", "Field_6", "Field_7"]
            labels = [tkinter.Label(dialog, text=label_texts[i]).grid(row=i, column=0, sticky=tkinter.NW) for i in range(8)]
            entries = [tkinter.Entry(dialog) for i in range(0, 8)]
            for i in range(len(entries)): entries[i].grid(row=i, column=1, sticky=tkinter.NW)
            for i in range(len(entries)): entries[i].insert(0, str(t.behaviours[block_id][i]))
            def apply_changes():
                values = [entry.get() for entry in entries]
                if primary: self.primary_tileset.behaviours[block_id] = values
                else: self.secondary_tileset.behaviours[block_id] = values
            tkinter.Button(dialog, text="Apply", command=apply_changes).grid(row=8, column=1, sticky=tkinter.NW)

        context_menu_tsp.add_command(label="Properties", command=lambda: create_properties_dialog(True))
        context_menu_tss.add_command(label="Properties", command=lambda: create_properties_dialog(False))
        def context_menu_popup(event, primary):
            if not self.primary_tileset or not self.secondary_tileset: return
            if self.primary_tileset.image.empty or self.secondary_tileset.image.empty: return
            block_selection_b_press(event, primary)
            if primary: context_menu_tsp.post(event.x_root, event.y_root)
            else: context_menu_tss.post(event.x_root, event.y_root)

        self.blocks_tsp_canvas.bind("<ButtonPress-1>", lambda e: block_selection_b_press(e, True))
        self.blocks_tss_canvas.bind("<ButtonPress-1>", lambda e: block_selection_b_press(e, False))
        self.blocks_tsp_canvas.bind("<ButtonPress-3>", lambda e: context_menu_popup(e, True))
        self.blocks_tss_canvas.bind("<ButtonPress-3>", lambda e: context_menu_popup(e, False))
        self.blocks_tsp_canvas.pack(anchor=tkinter.NW)
        self.blocks_tss_canvas.pack(anchor=tkinter.NW)


        #Setup the two canvases for block editing (one for tsp and tss)
        #-------------------------------------------------------------------
        def save_block_to_build(primary):
            if not self.primary_tileset or not self.secondary_tileset: return
            if self.primary_tileset.image.empty or self.secondary_tileset.image.empty: return
            data = self.block_to_build_tsp if primary else self.block_to_build_tss
            t = self.primary_tileset if primary else self.secondary_tileset
            selected_block = self.selected_block_tsp if primary else self.selected_block_tss
            t.blocks[selected_block] = [(tile | (vflip << 10) | (hflip << 11) | (pal << 12))for tile, hflip, vflip, pal in data]
            self.refresh_block_in_selection(primary, selected_block)
            raw_pil_image = self.blocks_tsp_raw_pil_image if primary else self.blocks_tss_raw_pil_image
            if primary:
                self.blocks_tsp_raw_image = ImageTk.PhotoImage(raw_pil_image)
                self.blocks_tsp_canvas.delete(self.blocks_tsp_canvas_image)
                self.blocks_tsp_canvas.create_image(1, 1, image=self.blocks_tsp_raw_image, anchor=tkinter.NW)
            else:
                self.blocks_tss_raw_image = ImageTk.PhotoImage(raw_pil_image)
                self.blocks_tss_canvas.delete(self.blocks_tss_canvas_image)
                self.blocks_tss_canvas.create_image(1, 1, image=self.blocks_tss_raw_image, anchor=tkinter.NW)
            self.refresh_block_selection_rectangle(primary)
        bedit_frame_tsp, bedit_frame_tss = tkinter.Frame(self.frame_tsp), tkinter.Frame(self.frame_tss)
        bedit_frame_tsp.grid(row=1, sticky=tkinter.NW), bedit_frame_tss.grid(row=1, sticky=tkinter.NW)
        bt_block_edit_tsp_expl, bt_block_edit_tss_expl = tkinter.Button(bedit_frame_tsp, text="Save", command=lambda: save_block_to_build(True)), tkinter.Button(bedit_frame_tss, text="Save", command=lambda: save_block_to_build(False))
        bt_block_edit_tsp_expl.grid(row=2, column=0, sticky=tkinter.NW)
        bt_block_edit_tss_expl.grid(row=2, column=0, sticky=tkinter.NW)
        self.lb_head_block_edit_tsp, self.lb_head_block_edit_tss = tkinter.Label(bedit_frame_tsp, text="Block #???"), tkinter.Label(bedit_frame_tss, text="Block #???")
        self.lb_head_block_edit_tsp.grid(row=0, column=0, sticky=tkinter.NW)
        self.lb_head_block_edit_tss.grid(row=0, column=0, sticky=tkinter.NW)
        self.block_tsp_edit_canvas = tkinter.Canvas(bedit_frame_tsp, width=8 * CANVAS_ZOOM * 2, height = 8 * CANVAS_ZOOM * 4, background="black")
        self.block_tss_edit_canvas = tkinter.Canvas(bedit_frame_tss, width=8 * CANVAS_ZOOM * 2, height = 8 * CANVAS_ZOOM * 4, background="black")
        self.block_tsp_edit_raw_pil_image = PImage.new("RGBA", (8 * CANVAS_ZOOM * 2, 8 * CANVAS_ZOOM * 4), "black")
        self.block_tss_edit_raw_pil_image = PImage.new("RGBA", (8 * CANVAS_ZOOM * 2, 8 * CANVAS_ZOOM * 4), "black")
        self.block_tsp_edit_raw_image = ImageTk.PhotoImage(self.block_tsp_edit_raw_pil_image)
        self.block_tss_edit_raw_image = ImageTk.PhotoImage(self.block_tss_edit_raw_pil_image)
        self.block_tsp_edit_canvas_image = self.block_tsp_edit_canvas.create_image(1, 1, image=self.block_tsp_edit_raw_image, anchor=tkinter.NW)
        self.block_tss_edit_canvas_image = self.block_tss_edit_canvas.create_image(1, 1, image=self.block_tss_edit_raw_image, anchor=tkinter.NW)
        self.block_tsp_edit_canvas.grid(row=1, column=0, sticky=tkinter.NW)
        self.block_tss_edit_canvas.grid(row=1, column=0, sticky=tkinter.NW)
        def edit_canvas_b_press(event, primary):
            if not self.primary_tileset or not self.secondary_tileset: return
            block_to_build = self.block_to_build_tsp if primary else self.block_to_build_tss
            x, y = int(event.x / (8 * CANVAS_ZOOM)), int(event.y / (8 * CANVAS_ZOOM))
            for j in range(len(self.selected_tiles)): #Iterate through every line
                if y + j < 4: #Selection only holds 4 tiles
                    for i in range(len(self.selected_tiles[j])):
                        if x + i < 2:
                            block_to_build[2 * (y + j) + x + i] = self.selected_tiles[j][i]
            self.refresh_block_edit(primary)
        def edit_canvas_b3_press(event, primary):
            if not self.primary_tileset and self.secondary_tileset: return
            x, y = int(event.x / (8 * CANVAS_ZOOM)), int(event.y / (8 * CANVAS_ZOOM))
            if primary: self.block_to_build_tsp_selected_area = x, y, x + 1, y + 1
            else: self.block_to_build_tss_selected_area = x, y, x + 1, y + 1
            self.refresh_tile_selection_by_block_area(primary)
            #Change the tileset view to fit this selection
        def edit_canvas_b3_motion(event, primary):
            if not self.primary_tileset and self.secondary_tileset: return
            x1, y1 = int(event.x / (8 * CANVAS_ZOOM)) + 1, int(event.y / (8 * CANVAS_ZOOM)) + 1
            if x1 > 2: x1 = 2
            if y1 > 4: y1 = 4
            x0, y0, _, _ = self.block_to_build_tsp_selected_area if primary else self.block_to_build_tss_selected_area
            dx = x1 - x0
            dy = y1 - y0
            if dx <= 0 or dy <= 0: return
            if primary: self.block_to_build_tsp_selected_area = x0, y0, x1, y1
            else: self.block_to_build_tss_selected_area = x0, y0, x1, y1
            self.refresh_tile_selection_by_block_area(primary)
        def edit_canvas_b3_release(event, primary):
            x0, y0, x1, y1 = self.block_to_build_tsp_selected_area if primary else self.block_to_build_tss_selected_area
            if x1 - x0 == 1 and y1 - y0 == 1:   
                tile, hflip, vflip, pal = self.block_to_build_tsp[2 * y0 + x0] if primary else self.block_to_build_tss[2 * y0 + x0]
                tx, ty = tile & 0xF, tile >> 4
                self.tileset_selected_area = tx, ty, tx + 1, ty + 1
                self.tileset_pal_combobox.set(str(pal))
                self.tile_selection_hflip_box.set(hflip, ignore_trace=True)
                self.tile_selection_vflip_box.set(vflip, ignore_trace=True)
                self.tileset_canvas.delete(self.tileset_canvas_selection_rectangle)
                self.tileset_canvas_selection_rectangle = self.tileset_canvas.create_rectangle(tx * 8 * CANVAS_ZOOM, ty * 8 * CANVAS_ZOOM, (tx + 1) * 8 * CANVAS_ZOOM, (ty + 1) * 8 * CANVAS_ZOOM, fill="", outline="red")
                slider_top = ty / 64
                if slider_top > (2 / 3): slider_top = 2 / 3
                slider_bottom = slider_top + (1 / 3)
                self.tileset_canvas_bar_v.set(slider_top, slider_bottom)
                self.tileset_canvas.yview("moveto", slider_top)

        self.block_tsp_edit_canvas.bind("<B1-Motion>", lambda e: edit_canvas_b_press(e, True))
        self.block_tss_edit_canvas.bind("<B1-Motion>", lambda e: edit_canvas_b_press(e, False))
        self.block_tsp_edit_canvas.bind("<ButtonPress-3>", lambda e: edit_canvas_b3_press(e, True))
        self.block_tss_edit_canvas.bind("<ButtonPress-3>", lambda e: edit_canvas_b3_press(e, False))
        self.block_tsp_edit_canvas.bind("<B3-Motion>", lambda e: edit_canvas_b3_motion(e, True))
        self.block_tss_edit_canvas.bind("<B3-Motion>", lambda e: edit_canvas_b3_motion(e, False))
        self.block_tsp_edit_canvas.bind("<ButtonRelease-3>", lambda e: edit_canvas_b3_release(e, True))
        self.block_tss_edit_canvas.bind("<ButtonRelease-3>", lambda e: edit_canvas_b3_release(e, False))

        bpframe_tsp, bpframe_tss = tkinter.Frame(self.frame_tsp), tkinter.Frame(self.frame_tss)
        bpframe_tsp.grid(row=1, column=1, sticky=tkinter.NW), bpframe_tss.grid(row=1, column=1, sticky=tkinter.NW)


        tkinter.Button(bpframe_tsp, text="Update blocks (long computing time...)", command=lambda: self.refresh_block_selection(True)).grid(row=0, column=0, columnspan=3, sticky=tkinter.NW)
        tkinter.Button(bpframe_tss, text="Update blocks (long computing time...)", command=lambda: self.refresh_block_selection(False)).grid(row=0, column=0, columnspan=3, sticky=tkinter.NW)
    
        #Setup two widgets for data editing
        #-------------------------------------------------------------------
        tkinter.Label(bpframe_tsp, text="Symbol").grid(row=1, column=0, sticky=tkinter.NW)
        tkinter.Label(bpframe_tss, text="Symbol").grid(row=1, column=0, sticky=tkinter.NW)
        self.tsp_sym_combobox = tkinterx.Combobox(bpframe_tsp, values=self.project.get_tileset_symbols(_sorted=True), state="readonly")
        self.tss_sym_combobox = tkinterx.Combobox(bpframe_tss, values=self.project.get_tileset_symbols(_sorted=True), state="readonly")
        self.tsp_sym_combobox.grid(row=1, column=1, sticky=tkinter.NW)
        self.tss_sym_combobox.grid(row=1, column=1, sticky=tkinter.NW)
        tkinter.Label(bpframe_tsp, text="Gfx").grid(row=2, column=0, sticky=tkinter.NW)
        tkinter.Label(bpframe_tss, text="Gfx").grid(row=2, column=0, sticky=tkinter.NW)
        self.tsp_gfx_combobox = tkinterx.Combobox(bpframe_tsp, values=self.project.get_image_symbols(_sorted=True), state="readonly")
        self.tss_gfx_combobox = tkinterx.Combobox(bpframe_tss, values=self.project.get_image_symbols(_sorted=True), state="readonly")
        self.tsp_gfx_combobox.grid(row=2, column=1, sticky=tkinter.NW)
        self.tss_gfx_combobox.grid(row=2, column=1, sticky=tkinter.NW)
        tkinter.Label(bpframe_tsp, text="Anim. init.").grid(row=3, column=0, sticky=tkinter.NW)
        tkinter.Label(bpframe_tss, text="Anim. init.").grid(row=3, column=0, sticky=tkinter.NW)
        self.tsp_init_func_entry = tkinterx.Entry(bpframe_tsp)
        self.tss_init_func_entry = tkinterx.Entry(bpframe_tss)
        self.tsp_init_func_entry.grid(row=3, column=1, sticky=tkinter.NW)
        self.tss_init_func_entry.grid(row=3, column=1, sticky=tkinter.NW)
        self.tsp_error_label = tkinter.Label(bpframe_tsp, fg="red")
        self.tsp_error_label.grid(row=4, column=0, sticky=tkinter.NW, columnspan=2)
        self.tss_error_label = tkinter.Label(bpframe_tss, fg="red")
        self.tss_error_label.grid(row=4, column=0, sticky=tkinter.NW, columnspan=2)
        tkinter.Button(bpframe_tsp, text="Export in selected palette", command = lambda: self.export_tileset_gfx(True)).grid(row=5, column=0, columnspan=3, sticky=tkinter.NW)
        tkinter.Button(bpframe_tss, text="Export in selected palette", command = lambda: self.export_tileset_gfx(False)).grid(row=5, column=0, columnspan=3, sticky=tkinter.NW)

        def set_anim_init_func(primary, symbol):
            if primary and self.primary_tileset:
                self.primary_tileset.init_func = symbol
            elif not primary and self.secondary_tileset:
                self.secondary_tileset.init_func = symbol

        def set_gfx(primary, symbol):
            img = self.project.get_image_path(symbol)
            if not img:
                return
            if primary and self.primary_tileset:
                try:
                    self.primary_tileset.gfx = symbol
                    self.primary_tileset.load_image_file(img)
                    self.refresh_all()
                    self.err(primary) #clear err
                except Exception as e:
                    #messagebox.showerror(title="Error while loading gfx", message="Gfx symbol '" + symbol + "' could not be loaded: " + str(e))
                    self.err(primary, "Gfx symbol '" + symbol + "' could not be loaded: " + str(e))
                    #self.root.grab_set()
                    self.refresh_widgets(True), self.refresh_widgets(False)
            elif not primary and self.secondary_tileset:
                try:
                    self.secondary_tileset.gfx = symbol
                    self.secondary_tileset.load_image_file(img)
                    self.refresh_all()
                    self.err(primary)
                except Exception as e:
                    #messagebox.showerror(title="Error while loading gfx", message="Gfx symbol '" + symbol + "' could not be loaded: " + str(e))
                    self.err(primary, "Gfx symbol '" + symbol + "' could not be loaded: " + str(e))
                    #self.root.grab_set()
                    self.refresh_widgets(True), self.refresh_widgets(False)

        def reassing_tileset(primary, symbol):
            ts = self.project.get_tileset(symbol)
            if not ts:
                return self.err(primary, "Tileset symbol '" + symbol + "' is not defined!")
                #return messagebox.showerror(title="Tileset symbol not defined", message="Tileset symbol '" + symbol + "' is not defined!")
            #print(primary, ts.is_primary, ts.symbol)
            if primary:
                if not ts.is_primary:
                    self.primary_tileset = None
                    self.refresh_all()
                    self.err(primary, "Unable to assing tileset symbol '" + symbol +  "' because it does not match the tileset type")
                else: self.assign_tileset(ts)
            else:
                if ts.is_primary:
                    self.secondary_tileset = None
                    self.refresh_all()
                    self.err(primary, "Unable to assing tileset symbol '" + symbol +  "' because it does not match the tileset type")
                else: self.assign_tileset(ts)
                
        self.tsp_init_func_entry.add_callback(lambda: set_anim_init_func(True, self.tsp_init_func_entry.get()))
        self.tss_init_func_entry.add_callback(lambda: set_anim_init_func(False, self.tss_init_func_entry.get()))
        self.tsp_gfx_combobox.add_callback(lambda: set_gfx(True, self.tsp_gfx_combobox.get()))
        self.tss_gfx_combobox.add_callback(lambda: set_gfx(False, self.tss_gfx_combobox.get()))
        self.tsp_sym_combobox.add_callback(lambda: reassing_tileset(True, self.tsp_sym_combobox.get()))
        self.tss_sym_combobox.add_callback(lambda: reassing_tileset(False, self.tss_sym_combobox.get()))

    def refresh_tile_selection_by_block_area(self, primary):
        """ Refreshes the tile selection by the selected area inside a block (tsp or tss)"""
        if not self._can_draw(): return
        x0, y0, x1, y1 = self.block_to_build_tsp_selected_area if primary else self.block_to_build_tss_selected_area
        blocks = self.block_to_build_tsp if primary else self.block_to_build_tss
        self.selected_tiles = [[blocks[2 * (y0 + j) + x0 + i] for i in range(x1 - x0)] for j in range(y1 - y0)]
        self.refresh_tile_selection()


    def refresh_tile_selection_by_tileset_area(self):
        """ Refreshes the tile selection by the selected area inside the tileset widget"""
        if not self._can_draw(): return
        x0, y0, x1, y1 = self.tileset_selected_area
        hflip = self.tile_selection_hflip_box.get()
        vflip = self.tile_selection_vflip_box.get()
        self.selected_tiles = [[(16 * (y0 + j) + x0 + i, hflip, vflip, int(self.tileset_pal_combobox.get())) for i in range(x1 - x0)] for j in range(y1 - y0)]
        self.refresh_tile_selection()
        self.tileset_canvas.delete(self.tileset_canvas_selection_rectangle)
        self.tileset_canvas_selection_rectangle = self.tileset_canvas.create_rectangle(x0 * 8 * CANVAS_ZOOM, y0 * 8 * CANVAS_ZOOM, x1 * 8 * CANVAS_ZOOM, y1 * 8 * CANVAS_ZOOM, fill="", outline="red")

    def refresh_block_edit(self, primary):
        """ Refreshes the block edit canvas for a given tileset (tss or tsp) and all"""
        if not self._can_draw(): return
        t = self.primary_tileset if primary else self.secondary_tileset
        raw_pil_image = self.block_tsp_edit_raw_pil_image if primary else self.block_tss_edit_raw_pil_image
        canvas_image = self.block_tsp_edit_canvas_image if primary else self.block_tss_edit_canvas_image
        canvas = self.block_tsp_edit_canvas if primary else self.block_tss_edit_canvas
        tiles = self.block_to_build_tsp if primary else self.block_to_build_tss
        for i in range(8):
            tile, hflip, vflip, pal_id = tiles[i]
            x, y = i & 1, (i >> 1)
            if pal_id > 12: continue
            tile_image = self.get_tile(tile, hflip, vflip, pal_id).resize((8 * CANVAS_ZOOM, 8 * CANVAS_ZOOM))
            raw_pil_image.paste(tile_image, (x * 8 * CANVAS_ZOOM, y * 8 * CANVAS_ZOOM))
        if primary: self.block_tsp_edit_raw_image = ImageTk.PhotoImage(raw_pil_image)
        else: self.block_tss_edit_raw_image = ImageTk.PhotoImage(raw_pil_image)
        raw_image = self.block_tsp_edit_raw_image if primary else self.block_tss_edit_raw_image
        canvas.delete(canvas_image)
        canvas.create_image(1, 1, image=raw_image, anchor=tkinter.NW)


    def refresh_block_in_selection(self, primary, i):
        """ Refreshes a single block (index i) in the canvas for a given tileset (tss or tsp) and all canvases linked to it (cascade)"""
        if not self._can_draw(): return
        t = self.primary_tileset if primary else self.secondary_tileset        
        block_x, block_y = i % 8, int(i / 8)
        block = t.blocks[i]
        """
        agb_image = self.blocks_tsp_agb_image if primary else self.blocks_tss_agb_image
        for k in range(8):
            tile_fields = block[k] #The tilemap data for this block
            tile = tile_fields & 0x3FF #The tile id
            t_src = self.primary_tileset if tile < 0x280 else self.secondary_tileset
            if tile >= 0x280: tile -= 0x280 #The secondary tilesets holds tiles >= 0x280
            hflip = True if tile_fields & 0x400 else False
            vflip = True if tile_fields & 0x800 else False
            pal_id = (tile_fields >> 12) & 0xF
            #Find the right palette
            if pal_id < 7: colors = self.primary_tileset.palettes[pal_id].colors
            else: colors = self.secondary_tileset.palettes[pal_id - 7].colors
            j = k & 3 #The index inside the 2x2 block (lower or higher)
            tile_y, tile_x = tile >> 4, tile & 0xF
            agb_image.overlay(t_src.get_agb_image(), (16 * (block_x + x_inner), 16 * (block_y + y_inner)), (tile_x * 8, tile_y * 8, (tile_x + 1) * 8, (tile_y + 1) * 8), channel = k >= 4)
        """

        raw_pil_image = self.blocks_tsp_raw_pil_image if primary else self.blocks_tss_raw_pil_image
        #print(raw_pil_image.size, block_x, block_y, block)
        #Iterate through all 8 members of a block (2x2 lower, 2x2 higher)
        for k in range(8): 
            tile_fields = block[k] #The tilemap data for this block
            tile = tile_fields & 0x3FF #The tile id
            vflip = True if tile_fields & 0x400 else False
            hflip = True if tile_fields & 0x800 else False
            pal_id = (tile_fields >> 12) & 0xF
            if pal_id > 12: continue
            j = k & 3 #The index inside the 2x2 block (lower or higher)
            tile_image = self.get_tile(tile, hflip, vflip, pal_id).resize((8 * CANVAS_ZOOM, 8 * CANVAS_ZOOM))
            x_inner, y_inner =  j & 1, j >> 1
            if k < 4: #Paste without alpha channel
                raw_pil_image.paste(tile_image, (CANVAS_ZOOM * (16 * block_x + 8 * x_inner), CANVAS_ZOOM * (16 * block_y + 8 * y_inner)))
            else: #Paste with alpha channel
                raw_pil_image.paste(tile_image, (CANVAS_ZOOM * (16 * block_x + 8 * x_inner), CANVAS_ZOOM * (16 * block_y + 8 * y_inner)), tile_image)
            
    

    def refresh_block_selection(self, primary):
        """ Refreshes a block canvas for a given tileset (tss or tsp) and all canvases linked to it (cascade)"""
        if not self._can_draw(): return
        t = self.primary_tileset if primary else self.secondary_tileset
        #Refresh all blocks, which causes a lot of computing time...

        print("Computing blocks, this might take a while...")
        t0 = time.time()
        """
        Threading does not work here, since PIL images are locked
        threads = [None] * BLOCK_RENDERING_JOBS
        #Define an inner method that renders all blocks i where i % module == 0 (we split rendering to seperate threads)
        def _render_blocks_module(module):
            for i in range(module, len(t.blocks), BLOCK_RENDERING_JOBS):
                self.refresh_block_in_selection(primary, i)
        for j in range(BLOCK_RENDERING_JOBS):
            threads[j] = Thread(target=_render_blocks_module, args=(j, ))
            threads[j].start()
        for thread in threads:
            thread.join()
        """
        for i in range(len(t.blocks)):
            self.refresh_block_in_selection(primary, i)

        t1 = time.time()
        #print("Done! Took me ", t1-t0, "s")
        
        raw_pil_image = self.blocks_tsp_raw_pil_image if primary else self.blocks_tss_raw_pil_image
        if primary:
            self.blocks_tsp_raw_image = ImageTk.PhotoImage(raw_pil_image)
            self.blocks_tsp_canvas.delete(self.blocks_tsp_canvas_image)
            self.blocks_tsp_canvas.create_image(1, 1, image=self.blocks_tsp_raw_image, anchor=tkinter.NW)
        else:
            self.blocks_tss_raw_image = ImageTk.PhotoImage(raw_pil_image)
            self.blocks_tss_canvas.delete(self.blocks_tss_canvas_image)
            self.blocks_tss_canvas.create_image(1, 1, image=self.blocks_tss_raw_image, anchor=tkinter.NW)
        self.refresh_block_selection_rectangle(primary)
        
    def refresh_block_selection_rectangle(self, primary):
        """ Redraws the red rectangle for block selection (this needs to be called seperatly when you choose a tile or on other instances) """
        if not self._can_draw(): return
        if primary:
            x, y = self.selected_block_tsp & 7, self.selected_block_tsp >> 3
            self.blocks_tsp_canvas.delete(self.blocks_tsp_canvas_rectangle)
            self.blocks_tsp_canvas_rectangle = self.blocks_tsp_canvas.create_rectangle(x * 16 * CANVAS_ZOOM, y * 16 * CANVAS_ZOOM, (x + 1) * 16 * CANVAS_ZOOM, (y + 1) * 16 * CANVAS_ZOOM, fill="", outline="red")
        else:
            x, y = self.selected_block_tss & 7, self.selected_block_tss >> 3
            self.blocks_tss_canvas.delete(self.blocks_tss_canvas_rectangle)
            self.blocks_tss_canvas_rectangle = self.blocks_tss_canvas.create_rectangle(x * 16 * CANVAS_ZOOM, y * 16 * CANVAS_ZOOM, (x + 1) * 16 * CANVAS_ZOOM, (y + 1) * 16 * CANVAS_ZOOM, fill="", outline="red")
            

    def refresh_title(self):
        """ Refreshes the title of the gui """
        title = "pytileset - primary tileset: "
        if self.primary_tileset:
            title += self.primary_tileset.symbol
        else:
            title += "not loaded"
        title += ", secondary tileset; "
        if self.secondary_tileset:
            title += self.secondary_tileset.symbol
        else:
            title += "not loaded"
        self.root.wm_title(title)

        
    def refresh_tile_selection(self):
        """ Refreshes the canvas for the selected tiles """
        if not self._can_draw(): return
        #self.tileset_canvas.coords(self.tileset_canvas_selection_rectangle, x0 * 8 * CANVAS_ZOOM, y0 * 8 * CANVAS_ZOOM, x1 * 8 * CANVAS_ZOOM, y1 * 8 * CANVAS_ZOOM)
        #Create a new image for the tile selection
        merged_img = PImage.new("RGBA", (8 * CANVAS_ZOOM * 2, 8 * CANVAS_ZOOM * 4))
        #Collect tiles
        for y in range(len(self.selected_tiles)): #Iterate through every line of selection
            for x in range(len(self.selected_tiles[y])): #Iterate through every tile
                tile, hflip, vflip, pal = self.selected_tiles[y][x]
                tile_image = self.get_tile(tile, hflip, vflip, pal).resize((8 * CANVAS_ZOOM, 8 * CANVAS_ZOOM))
                merged_img.paste(tile_image, (x * 8 * CANVAS_ZOOM, y * 8 * CANVAS_ZOOM))

        self.tile_selection_canvas.delete(self.tile_selection_canvas_image)
        self.tile_selection_raw_image = ImageTk.PhotoImage(image=merged_img)
        self.tile_selection_canvas.create_image(1, 1, image=self.tile_selection_raw_image, anchor=tkinter.NW)
    
    def get_tile(self, tile, hflip, vflip, pal):
        """ Gets a tile form one of the two resources (tsp, tss)"""
        if not self._can_draw(): return
        if not self.primary_tileset or not self.secondary_tileset: return
        if tile < 0x280:
            t_src = self.primary_tileset
        else:
            t_src = self.secondary_tileset
            tile -= 0x280
        if pal < 7: colors = self.primary_tileset.palettes[pal]
        else: colors = self.secondary_tileset.palettes[pal - 7]
        return t_src.get_tile(tile, colors, hflip, vflip)
        

    def refresh_tileset_canvas(self, refresh_tile_selection=True):
        """ Refreshes the tileset canvas and all canvases linked to its appearance (cascade)"""
        if not self._can_draw(): return
        if not self.primary_tileset or not self.secondary_tileset: return
        if self.primary_tileset.image.empty or self.secondary_tileset.image.empty: return
        pal_id = int(self.tileset_pal_combobox.get())
        if pal_id < 7:
            colors = self.primary_tileset.palettes[pal_id]
        else:
            colors = self.secondary_tileset.palettes[pal_id - 7]
        tsp_img = self.primary_tileset.get_image(colors, 16, 40)
        tss_img = self.secondary_tileset.get_image(colors, 16, 24)
        merged_img = PImage.new("RGBA", (128, 512))
        merged_img.paste(tsp_img.convert("RGBA"), (0, 0))
        merged_img.paste(tss_img.convert("RGBA"), (0, 320))
        merged_img = merged_img.resize((CANVAS_WIDTH, CANVAS_HEIGHT)) #Double the view for a better mapping expierience
        merged_img = merged_img.convert(mode="RGBA")
        self.tileset_raw_pil_image = merged_img
        self.tileset_canvas.delete(self.tileset_canvas_image)
        self.tileset_raw_image = ImageTk.PhotoImage(image=merged_img)
        self.tileset_canvas.create_image(1, 1, image=self.tileset_raw_image, anchor=tkinter.NW)

        #Refresh tile selection and both block canvases (those may have changed as well)
        if refresh_tile_selection: self.refresh_tile_selection()

    def open_pal(self, path, pal_id):
        """ Reads a palette file into a certain palette slot on a tileset and refreshes the canvas """
        if not self._can_draw(): return
        primary = False
        if pal_id < 7: primary = True
        else: pal_id -= 7
        t = self.primary_tileset if primary else self.secondary_tileset
        if not path or not len(path): return
        if not t: return
        try:
            t.palettes[pal_id] = palette.from_file(path)
        except Exception as e:
            messagebox.showerror("Could not read palette file", "Could not read palette file :" + str(e))
            return
        self.refresh_tileset_canvas()
        
        

    def open_pal_dialoge(self):
        """ Callback to open a palette and use it for tsp or tss """
        if not self._can_draw(): return
        pal_id = int(self.tileset_pal_combobox.get())
        options = {
            "initialdir" : os.getcwd(),
            "filetypes" : [("4bpp portable network graphics", ".png")],
            "title" : "Import palette",
            "parent" : self
        }
        path = filedialog.askopenfilename(**options)
        self.open_pal(path, pal_id)


    def export_tileset_gfx(self, primary):
        """ Exports the the tilesets gfx in the currently
        selected palette by overwriting the png linked to
        the gfx"""
        t = self.primary_tileset if primary else self.secondary_tileset
        if not t: return

        # Get the palette
        pal_id = int(self.tileset_pal_combobox.get())
        if pal_id < 7:
            if not self.primary_tileset: return
            pal = self.primary_tileset.palettes[pal_id]
        else:
            if not self.secondary_tileset: return
            pal = self.secondary_tileset.palettes[pal_id - 7]
        
        # Group the flat palette into RGB triples
        rgbs = list(zip(pal[0::3], pal[1::3], pal[2::3]))
        print(len(rgbs))
        
        path = self.project.get_image_path(t.gfx)
       
        # Use the png method to generate 4bpp pngs
        width, height, pixels, _ = png.Reader(path).read()

        # Extract pixel information
        pixels = list(pixels)

        with open(path, "wb") as f:
            png.Writer(width, height, palette=rgbs, bitdepth=4).write(f, pixels)
        print("Sucessfully exported tileset gfx with palette", pal_id, "to", path)
        


    """
    def open_pal_path(self, primary):
        return #Deprecated
         Callback to open a palette by its path and use it for tsp or tss 
        t = self.primary_tileset if primary else self.secondary_tileset
        pal_id = int(self.tsp_pal_dropdown.get()) if primary else int(self.tss_pal_dropdown.get()) - 7
        if not t: return
        widget_entry = self.tsp_pal_entry if primary else self.tss_pal_entry
        path = widget_entry.get()
        self.open_pal(path, primary, pal_id)

    def open_image_dialoge(self, primary):
        return #Deprecated
        t = self.primary_tileset if primary else self.secondary_tileset
        if not t: return
        options = {}
        if t.image.key:
            options["initialdir"] = t.image.key
        options["filetypes"] = [('portable network graphics', '.png')]
        options["title"] = "Load image"
        options["parent"] = self
        path = filedialog.askopenfilename(**options)
        self.load_image(path, primary)
        self.refresh_tileset_canvas()
    def open_image_path(self, primary):
        return #Deprecated
        t = self.primary_tileset if primary else self.secondary_tileset
        if not t: return
        widget_entry = self.tsp_img_entry if primary else self.tss_img_entry
        path = widget_entry.get()
        try:
            t.load_image_file(os.path.relpath(path))
        except Exception as e:
            messagebox.showerror("Could not open image file", "Invalid image file or path " + path + "\n" + str(e))



    def palette_widget_pal_selected(self):
        return #Deprecated
        #Todo change the displayed tileset in right palette
        self.palette_widget_entry_pal.delete(0, tkinter.END)
        self.palette_widget_entry_pal.insert(0, self.active_palette_key)



    def palette_widget_refresh(self):
        return #Deprecated
        self.palette_keys = [str(i + (0 if self.tileset.is_primary else 7)) + ": " + str(os.path.basename(self.tileset.palettes[i].key) if self.tileset.palettes[i].key else "null") for i in range(len(self.tileset.palettes))]
        self.active_palette_key.set(self.palette_keys[0])
        self.palette_widget_dropdown["menu"].delete(0, tkinter.END)
        for pkey in self.palette_keys:
            self.palette_widget_dropdown["menu"].add_command(label=pkey, command = tkinter._setit(self.active_palette_key, pkey, callback=self.palette_widget_pal_selected))
"""

    def assign_tileset(self, t):
        """ Assigns a new tileset to the current state of the gui"""
        print("assigning", t.symbol)
        if t.is_primary:
            self.primary_tileset = t
        else:
            self.secondary_tileset = t

        #Update the image browsing widget
        img = self.project.get_image_path(t.gfx)
        if not img:
            #messagebox.showerror("Gfx symbol not defined", message="Gfx symbol '" + t.gfx + "' is not defined!")
            #self.root.grab_set()
            self.err(t.is_primary, "Gfx symbol '" + t.gfx + "' is not defined!")
            return
        t.load_image_file(img)
        self.refresh_all()
        self.err(t.is_primary)

    def refresh_widgets(self, primary):
        """ Refreshes the widgets of a tileset (tsp or tss)"""
        sym, gfx, init_func = "", "", ""
        if primary and self.primary_tileset:
            sym = self.primary_tileset.symbol
            gfx = self.primary_tileset.gfx
            init_func = self.primary_tileset.init_func
        elif not primary and self.secondary_tileset:
            sym = self.secondary_tileset.symbol
            gfx = self.secondary_tileset.gfx
            init_func = self.secondary_tileset.init_func
        if primary:
            self.tsp_sym_combobox.set(sym, untraced=True)
            self.tsp_gfx_combobox.set(gfx, untraced=True)
            self.tsp_init_func_entry.set(init_func, untraced=True)
        else:
            self.tss_sym_combobox.set(sym, untraced=True)
            self.tss_gfx_combobox.set(gfx, untraced=True)
            self.tss_init_func_entry.set(init_func, untraced=True)

    def refresh_all(self):
        self.refresh_widgets(True)
        self.refresh_widgets(False)
        if not self._can_draw(): return
        self.refresh_tileset_canvas()
        self.refresh_block_selection(True)
        self.refresh_block_selection(False)



    def file_new(self, primary):
        """ Creates a dialoge to create a new tileset file"""
        if primary and self.primary_tileset:
            if tkinter.messagebox.askyesno("Save changes to primary", "Do you want to save your changes to the primary tileset?"):
                if not self.file_save(primary): return #Save was unsucessfull, so we do not create a new file
        elif not primary and self.secondary_tileset:
            if tkinter.messagebox.askyesno("Save changes to secondary", "Do you want to save your changes to the secondary tileset?"):
                if not self.file_save(secondary): return #Save was unsucessfull, so we do not create a new file
        t = tileset.Tileset(primary)
        self.assign_tileset(t)
        self.refresh_title()


    def file_open(self, primary):
        """ Creates a dialoge to open a file and then opens the selected file """
        if primary and self.primary_tileset:
            if tkinter.messagebox.askyesno("Save changes to primary", "Do you want to save your changes to the primary tileset?"):
                if not self.file_save(primary): return #Save was unsucessfull, so we do not open a new file
        elif not primary and self.secondary_tileset:
            if tkinter.messagebox.askyesno("Save changes to secondary", "Do you want to save your changes to the secondary tileset?"):
                if not self.file_save(primary): return #Save was unsucessfull, so we do not open a new file
        options = {}
        if primary and self.primary_tileset and self.primary_tileset.path != tileset.PATH_UNSAFED:
            options["initialdir"] = self.primary_tileset.path
        elif not primary and self.secondary_tileset and self.secondary_tileset.path != tileset.PATH_UNSAFED:
            options["initialdir"] = self.secondary_tileset.path
        options["filetypes"] = [('pymap tileset files', '.pts')]
        options["title"] = "Open tileset file"
        options["parent"] = self
        path = filedialog.askopenfilename(**options)
        self.open_tileset_path(path, primary)

    def open_tileset(self, symbol):
        """ Opens a tileset by a symbol name """
        ts = self.project.get_tileset(symbol)
        if not ts:
            messagebox.showerror(title="Tileset not defined", message="The tileset symbol '" + symbol + "' is not defined!")
            #self.root.grab_set()
            return
        try:
            self.assign_tileset(ts)
        except Exception as e:
            messagebox.showerror(title="Tileset could not be assinged", message="Tileset symbol '" + symbol + "' could not be assigned: " + str(e))
            #self.root.grab_set()

    def open_tileset_path(self, path, primary):
        """ Opens a tileset by its path
        if not path or not len(path): return
        t = tileset.from_file(path)
        if t.is_primary != primary:
            messagebox.showerror("Wrong tileset type", "Selected tileset does not match the required type (primary, secondary)!")
            
            return
        self.assign_tileset(t)
        self.refresh_title()
        print("Tileset file '" + path + "' opened sucessfully")
        """
        pass #deprecated

    def load_image(self, symbol, primary):
        """ Loads a new image as the image file of the tileset 
        t = self.primary_tileset if primary else self.secondary_tileset
        if not t: return
        path = self.project.get_image_path(symbol)
        if not path or not len(path): return
        try: t.load_image_file(path)
        except Exception as e: tkinter.messagebox.showerror(title="Unable to load png", message="Unable to load png '" + path + "': " + str(e))"""
        pass #deprecated
        
            
    def file_save(self, primary):
        """ Creates a dialoge to save the current file """
        if primary and not self.primary_tileset:
            messagebox.showerror("No primary tileset", "There is not primary tileset to save!")
            return False
        elif not primary and not self.secondary_tileset:
            messagebox.showerror("No secondary tileset", "There is no secondary tileset to save!")
            return False
        t = self.primary_tileset if primary else self.secondary_tileset
        #Sync all not saved attributes

        if primary and self.primary_tileset:
            self.project.save_tileset(self.primary_tileset.symbol, self.primary_tileset, self.primary_tileset.path)
            print("Tileset file '" + self.primary_tileset.path + "'saved sucessfully")
        elif not primary and self.secondary_tileset:
            self.project.save_tileset(self.secondary_tileset.symbol, self.secondary_tileset, self.secondary_tileset.path)
            print("Tileset file '" + self.secondary_tileset.path + "'saved sucessfully")
        return True
    
    def _can_draw(self):
        """ Checks if the gui can draw any blocks (all data is instanciated correctly) """
        if not self.primary_tileset or not self.secondary_tileset: return False
        if not self.primary_tileset.image or not self.secondary_tileset.image: return False
        if self.primary_tileset.image.empty: return False
        if self.secondary_tileset.image.empty: return False
        return True

    def err(self, primary, estr=""):
        """ Shows an error message """
        if primary: self.tsp_error_label.config(text=estr)
        else: self.tss_error_label.config(text=estr)

