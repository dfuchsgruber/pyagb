import struct
import collections

class Agbrom:
    """
    This module provides a basic interface for Gameboy Advance ROMs.
    """
    def __init__(self, file_path, rom_start=0x8000000):
        """ Initializes the rom instance.
        
        Parameters:
        -----------
        file_path : string
            The path to the rom file.
        rom_start: int
            The offset in RAM where the rom will be loaded.
            Default = 0x8000000
        """
        with open(file_path, 'rb') as f:
            self.bytes = bytearray(f.read())

        self.rom_start = rom_start

        # Initialize views
        self.s8 = AgbromView(self, '<b')
        self.u8 = AgbromView(self, '<B')
        self.s16 = AgbromView(self, '<h')
        self.u16 = AgbromView(self, '<H')
        self.int = AgbromView(self, '<i')
        self.u32 = AgbromView(self, '<I')
        self.pointer = AgbromViewPointer(self, rom_start=self.rom_start)

    def __len__(self):
        return len(self.bytes)

    def find(self, pattern, alignment=0):
        """ Finds all occurences of a pattern in the rom.
        
        Parameters:
        -----------
        pattern : iterable
            The bytes to find.
        alignment : int
            Only occurences at offset that align with
            2 ** alignment will be considered.
            Default : 0
        
        Returns:
        -------
        offsets : list
            The offsets of the pattern.
        """
        position = -1
        bytes = bytearray(pattern)
        positions = []
        while True:
            position = self.bytes.find(bytes, position+1)
            if position >= 0:
                if position % (2 ** alignment) == 0:
                    positions.append(position)
                else:
                    print(f'Ignoring unaligned reference at {hex(position)}')
            else:
                break
        return positions

    def references(self, offset, alignment=2):
        """ Finds all occurences of a pointer to an offset
        in the rom.
        
        Parameters:
        -----------
        offset : int
            The offset of which pointers will be serached.
        alignment : int
            Only occurences at offset that align with
            2 ** alignment will be considered.
            Default : 2

        Returns:
        -------
        references : list
            Locations where references to offset were found.
        """
        bytes = bytearray(struct.pack('<I', offset + self.rom_start))
        return self.find(bytes, alignment=alignment)

class AgbromView:
    """ Provides a view on the Agbrom object """
    def __init__(self, rom, fmt):
        """
        Initializes a certian view (chars, shorts, etc.)
        on the Agbrom instance.

        The view can be accessed by subscribing:
        view.rom.int[offset] returns the int value at offset
        view.rom.int[offset : offset + k] returns k successive ints from
        offset, offset + 4, ..., offset + 4 * k
        view.rom.int[offset : offset + k : s] returns k successive ints
        from offset, offset + 1, ..., offset + k

        Parameters:
        -----------
        rom : Agbrom
            The agbrom to create a view on.
        fmt : string
            The format string for unpacking.
        """
        self.rom = rom
        self.fmt = fmt

    def get(self, offset):
        if offset < 0:
            offset += len(self.rom)
        return struct.unpack_from(self.fmt, self.rom.bytes, offset)[0]

    def set(self, offset, value):
        if offset < 0:
            offset += len(self.rom)
        struct.pack_into(self.fmt, self.rom.bytes, offset, value)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.get(offset) for offset in range(*key.indices(len(self.rom)))]
        elif isinstance(key, int):
            return self.get(key)
        elif isinstance(key, collections.Iterable):
            return [self.get(offset) for offset in key]
        else:
            raise RuntimeError(f'Unsupported indexing for AgbromView {key}. Only slices, iterables and integers are supported.')

    def __setitem__(self, key, item):
        if isinstance(key, int):
            if not isinstance(item, int):
                raise RuntimeError(f'Rom can only contain integer values!')
            self.set(key, item)
        else:
            # The key contains multiple incides
            if isinstance(key, slice):
                indices = range(*key.indices(len(self.rom)))
            elif isinstance(key, collections.Iterable):
                indices = key
            else:
                raise RuntimeError(f'Unsupported indexing for AgbromView {key}. Only slices, iterables and integers are supported.')
            if not isinstance(item, collections.Iterable):
                # Boradcast the value
                item = collections._repeat(item, len(indices))
            for offset, value in zip(indices, item):
                self.set(offset, value)


class AgbromViewPointer(AgbromView):
    """ Toplevel pointer view for the Agbrom object with same
    interface as AgbromView. """

    def __init__(self, rom, rom_start=0x8000000):
        """ 
        Parameters:
        -----------
        rom : Agbrom
            The agbrom to create a view on.
        rom_start : int
            The actual offset of the rom after being loaded into RAM.
            deafult : 0x8000000  (AGB)
        """
        super().__init__(rom, '<I')
        self.rom_start = rom_start

    def get(self, offset):
        result = super().get(offset)
        if result == 0:
            # Null pointer
            return None
        return result - self.rom_start

    def set(self, offset, item):
        if item is None:
            super().set(offset, 0)
        else:
            super().set(offset, item + self.rom_start)