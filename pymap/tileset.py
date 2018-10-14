import numpy as np
import json
import agb.palette
import agb.types
from . import backend

MAX_BLOCKS_PRIMARY = 0x200
MAX_BLOCKS_SECONDARY = 0x180
PALETTES_PRIMARY = 7
PALETTES_SECONDARY = 5

# Define a rgb color type
color_type = agb.types.BitfieldType('u16', [
    ('red', None, 5),
    ('blue', None, 5),
    ('green', None, 5)
])

# Define an array type to define a block tilemap
block_tilemap_type = agb.types.ArrayType(agb.types.u16, lambda _: 8)

# Define a behaviour type
behaviour_type = agb.types.BitfieldType('u32', [
    ('behaviour', None, 9),
    ('hm_usage', None, 5),
    ('field_2', None, 4),
    ('field_3', None, 6),
    ('encounter_type', None, 3),
    ('field_5', None, 2),
    ('field_6', None, 2),
    ('field_7', None, 1)
])

# Define a type for tilesets.
class TilesetType:
    """ Type class for map tilesets. """

    def __init__(self):
        """ Initializes the tileset type.
        
        Parameters:
        -----------
        is_primary : bool
            Wether the tileset is a primary tileset.
        """
        self.meta = agb.types.Structure([
            ('gfx_compressed', agb.types.u8),
            ('is_primary', agb.types.u8),
            ('field_2', agb.types.u8),
            ('field_3', agb.types.u8)
        ])
    
    def from_data(self, rom, offset, project, context, parents):
        """ Initializes all members. 
        
        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
            The rom to initialize the structure from
        offset : int
            The offset to initialize the structure from
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list of str
            The context in which the data got initialized
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.

        Returns:
        --------
        structure : dict
            The initialized structure
        offset : int
            The offset after all scalar members of the struct have been processed
        """
        structure = {}
        # Export metadata
        meta, offset = self.meta.from_data(rom, offset, project, context + ['meta'], parents + [structure])
        structure['meta'] = meta

        is_primary = bool(int(meta['is_primary']))
        gfx_compressed = bool(int(meta['gfx_compressed']))

        # Export gfx
        label, offset = backend.GfxPointerType(compressed=gfx_compressed).from_data(rom, offset, project, context + ['gfx'], parents + [structure])
        structure['gfx'] = label
        
        # Export palette
        palette_offset, offset = rom.pointer[offset], offset + 4
        palette_type = agb.types.ArrayType(color_type, lambda _: (PALETTES_PRIMARY * 16 if is_primary else PALETTES_SECONDARY * 16))
        palette, _ = palette_type.from_data(rom, palette_offset, project, context + ['palette'], parents + [structure])
        structure['palette'] = palette

        # Export blocks
        blocks_offset, offset = rom.pointer[offset], offset + 4
        blocks_type = agb.types.ArrayType(block_tilemap_type, lambda _: (MAX_BLOCKS_PRIMARY if is_primary else MAX_BLOCKS_SECONDARY))
        blocks, _ = blocks_type.from_data(rom, blocks_offset, project, context + ['blocks'], parents + [structure])
        structure['blocks'] = blocks

        # Export animation initializer
        animation_initialize, offset = rom.u32[offset], offset + 4
        structure['animation_initialize'] = animation_initialize

        # Export behaviours
        behaviours_offset, offset = rom.pointer[offset], offset + 4
        behaviours_type = agb.types.ArrayType(behaviour_type, lambda _: (MAX_BLOCKS_PRIMARY if is_primary else MAX_BLOCKS_SECONDARY))
        behaviours, _ = behaviours_type.from_data(rom, behaviours_offset, project, context + ['behaviours'], parents + [structure])
        structure['behaviours'] = behaviours

        return structure, offset

    def to_assembly(self, tileset, parents, label=None, alignment=2, global_label=False):
        """ Creates an assembly representation of the tileset.
        
        Parameters:
        -----------
        tileset : dict
            The tilesetr.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.
        label : string or None
            The label to export (only if not None).
        alignment : int or None
            The alignment of the structure if required
        global_label : bool
            If the label generated will be exported globally.
            Only relevant if label is not None.

        Returns:
        --------
        assembly : str
            The assembly representation of the tileset.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        blocks = []
        additional_blocks = []
        parents = parents + [tileset]

        # Create assembly for meta structure
        meta_assembly, meta_additional_blocks = self.meta.to_assembly(tileset['meta'], parents)
        blocks.append(meta_assembly)
        additional_blocks += meta_additional_blocks

        is_primary = bool(int(tileset['meta']['is_primary']))
        gfx_compressed = bool(int(tileset['meta']['gfx_compressed']))
        
        blocks.append(agb.types.pointer.to_assembly(tileset['gfx'], parents)[0])
    
        # Create assembly for palette
        palette_type = agb.types.ArrayType(color_type, lambda _: (PALETTES_PRIMARY * 16 if is_primary else PALETTES_SECONDARY * 16))
        palette_assembly, palette_additional_blocks = palette_type.to_assembly(tileset['palette'], parents, label='palette', alignment=2)
        additional_blocks.append(palette_assembly)
        additional_blocks += palette_additional_blocks
        blocks.append(agb.types.pointer.to_assembly('palette', parents)[0])

        # Create assembly for tilemap blocks
        blocks_type = agb.types.ArrayType(block_tilemap_type, lambda _: (MAX_BLOCKS_PRIMARY if is_primary else MAX_BLOCKS_SECONDARY))
        blocks_assembly, blocks_additional_blocks = blocks_type.to_assembly(tileset['blocks'], parents, label='blocks', alignment=2)
        additional_blocks.append(blocks_assembly)
        additional_blocks += blocks_additional_blocks
        blocks.append(agb.types.pointer.to_assembly('blocks', parents)[0])

        blocks.append(agb.types.pointer.to_assembly(tileset['animation_initialize'], parents)[0])

        # Create assembly for behaviours
        behaviours_type = agb.types.ArrayType(behaviour_type, lambda _: (MAX_BLOCKS_PRIMARY if is_primary else MAX_BLOCKS_SECONDARY))#
        behaviours_assembly, behaviours_additional_blocks = behaviours_type.to_assembly(tileset['behaviours'], parents, label='behaviours', alignment=2)
        additional_blocks.append(behaviours_assembly)
        additional_blocks += behaviours_additional_blocks
        blocks.append(agb.types.pointer.to_assembly('behaviours', parents)[0])

        return agb.types.label_and_align('\n'.join(blocks), label=label, alignment=alignment, global_label=global_label), additional_blocks

        

    def __call__(self, parents, is_primary=True, gfx_compressed=True):
        """ Initializes a new tileset.
        
        Parameters:
        -----------
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are generated depth-first.
        
        Returns:
        --------
        tileset : dict
            The new tileset.
        """
        tileset = {}
        parents += [tileset]
        tileset['meta'] = self.meta(parents)
        tileset['meta']['gfx_compressed'] = 1 if gfx_compressed else 0
        tileset['meta']['is_primary'] = 1 if is_primary else 0
        tileset['gfx'] = 0

        palette_type = agb.types.ArrayType(color_type, lambda _: (PALETTES_PRIMARY * 16 if is_primary else PALETTES_SECONDARY * 16))
        tileset['palette'] = palette_type(parents)

        blocks_type = agb.types.ArrayType(block_tilemap_type, lambda _: (MAX_BLOCKS_PRIMARY if is_primary else MAX_BLOCKS_SECONDARY))
        tileset['blocks'] = blocks_type(parents)

        tileset['animation_initialize'] = 0

        behaviours_type = agb.types.ArrayType(behaviour_type, lambda _: (MAX_BLOCKS_PRIMARY if is_primary else MAX_BLOCKS_SECONDARY))#
        tileset['behaviours'] = behaviours_type(parents)
        
        return tileset
