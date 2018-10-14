""" Module to model mapfooters. """

from . import event
import json
import agb.types
import pymap.backend

# Define the map block bitfield type
map_block_type = agb.types.BitfieldType('u16', [
    ('block_idx', None, 10),
    ('level', None, 6)
])

class FooterType:
    """ Class to model a mapfooter. """

    def from_data(self, rom, offset, proj, context, parents):
        """ Retrieves the footer from a rom.
        
        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
            The rom to retrieve the data from.
        offset : int
            The offset to retrieve the data from.
        proj : pymap.project.Project
            The pymap project to access e.g. constants.
        context : list of str
            The context in which the data got initialized
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.
        
        Returns:
        --------
        value : list of int or str
            The values at the given offset in rom associated with a constant string if possible.
        offset : int
            The offeset of the next bytes after the value was processed
        """
        context = context + ['footer']
        footer = {}
        parents = parents + [footer]

        width, offset = agb.types.u32.from_data(rom, offset, proj, context, parents)
        footer['width'] = width
        height, offset = agb.types.u32.from_data(rom, offset, proj, context, parents)
        footer['height'] = height

        # Retrieve the border offset, but do not initialize yet, since size is not availible right now
        border_offset, offset = agb.types.pointer.from_data(rom, offset, proj, context, parents)

        # Create an array type that fits the map size
        map_block_array_type = agb.types.ArrayType(map_block_type, lambda _: width * height)
        map_blocks_offset, offset = agb.types.pointer.from_data(rom, offset, proj, context, parents)
        map_blocks, _ = map_block_array_type.from_data(rom, map_blocks_offset, proj, context + ['blocks'], parents)
        footer['blocks'] = map_blocks

        # Export tilesets
        tileset_primary_offset, offset = agb.types.pointer.from_data(rom, offset, proj, context, parents)
        footer['tileset_primary'] = pymap.backend.tileset(rom, tileset_primary_offset, context + ['tileset_primary'])
        tileset_secondary_offset, offset = agb.types.pointer.from_data(rom, offset, proj, context, parents)
        footer['tileset_secondary'] = pymap.backend.tileset(rom, tileset_secondary_offset, context + ['tileset_secondary'])

        border_width, offset = agb.types.u8.from_data(rom, offset, proj, context, parents)
        footer['border_width'] = border_width
        border_height, offset = agb.types.u8.from_data(rom, offset, proj, context, parents)
        footer['border_height'] = border_height

        # Finally we can export the border
        border_type = agb.types.ArrayType(map_block_type, lambda _: border_width * border_height)
        border, _ = border_type.from_data(rom, border_offset, proj, context + ['border'], parents)
        footer['border'] = border

        field_16, offset = agb.types.u16.from_data(rom, offset, proj, context, parents)
        footer['field_16'] = field_16
        return footer, offset

    def to_assembly(self, footer, parents, label=None, alignment=None, global_label=False):
        """ Returns an assembly instruction line to export this scalar type.
        
        Parameters:
        -----------
        footer : dict
            The footer to export to assembly.
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
            The assembly representation of the footer.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        blocks, additional_blocks = [], []
        parents += footer

        width_assembly, width_additional_blocks = agb.types.u32.to_assembly(footer['width'], parents)
        blocks.append(width_assembly)
        additional_blocks += width_additional_blocks
        height_assembly, height_additional_blocks = agb.types.u32.to_assembly(footer['height'], parents)
        blocks.append(height_assembly)
        additional_blocks += height_additional_blocks

        border_type = agb.types.ArrayType(map_block_type, lambda _: int(footer['border_width']) * int(footer['border_height']))
        border_assembly, border_additional_blocks = border_type.to_assembly(footer['border'], parents, label='border', alignment=2)
        blocks.append(agb.types.pointer.to_assembly('border', parents)[0])
        additional_blocks.append(border_assembly)
        additional_blocks += border_additional_blocks

        map_block_array_type = agb.types.ArrayType(map_block_type, lambda _: int(footer['width']) * int(footer['height']))
        map_blocks_assembly, map_blocks_additional_blocks = map_block_array_type.to_assembly(footer['blocks'], parents, label='blocks', alignment=2)
        blocks.append(agb.types.pointer.to_assembly('blocks', parents)[0])
        additional_blocks.append(map_blocks_assembly)
        additional_blocks += map_blocks_additional_blocks
    
        blocks.append(agb.types.pointer.to_assembly(footer['tileset_primary'], parents)[0])
        blocks.append(agb.types.pointer.to_assembly(footer['tileset_secondary'], parents)[0])
        blocks.append(agb.types.u8.to_assembly(footer['border_width'], parents)[0])
        blocks.append(agb.types.u8.to_assembly(footer['border_height'], parents)[0])
        blocks.append(agb.types.u16.to_assembly(footer['field_16'], parents)[0])

        return agb.types.label_and_align('\n'.join(blocks), label=label, alignment=alignment, global_label=global_label), additional_blocks

    def __call__(self, parents):
        """ Initializes a empty footer.
        
        Parameters:
        -----------
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are generated depth-first.
        
        Returns:
        --------
        footer : dict
            The mapfooter.
        """
        footer = {
            'width' : 1,
            'height' : 1,
            'border_width' : 1,
            'border_height' : 1,
            'tileset_primary' : 0,
            'tileset_secondary' : 0,
            'field_16' : 0
        }
        parents += [footer]

        border_type = agb.types.ArrayType(map_block_type, lambda _: int(footer['border_width']) * int(footer['border_height']))
        footer['border'] = border_type(parents)

        map_block_array_type = agb.types.ArrayType(map_block_type, lambda _: int(footer['width']) * int(footer['height']))
        footer['blocks'] = map_block_array_type(parents)

        return footer
