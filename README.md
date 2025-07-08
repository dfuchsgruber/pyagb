# PyAGB - Game Boy Advance ROM Development Tools

A comprehensive Python toolkit for Game Boy Advance ROM hacking and development, with specialized support for Pokemon games. This project consists of two main packages: `pyagb` (core AGB data handling) and `pymap` (map editing and project management).

## Overview

PyAGB provides tools for:
- **ROM Data Manipulation**: Reading, writing, and processing Game Boy Advance ROM data
- **Map Editing**: Visual map editor with tileset, block, and event management
- **Project Management**: Organized workspace for ROM hacking projects
- **Asset Management**: Graphics, tilesets, palettes, and animations
- **Code Generation**: Assembly and C code generation from data structures

## Packages

### 🎮 pyagb - Core AGB Library
The foundational library for Game Boy Advance data processing.

**Features:**
- Low-level ROM data structures and types
- Image and palette processing (4-bit indexed color)
- LZ77 compression/decompression support
- String encoding/decoding with custom character maps
- Binary data model definitions
- Memory layout management

**Key Modules:**
- `agb.types` - Data type definitions for GBA structures
- `agb.image` - Image processing and conversion
- `agb.palette` - Color palette management
- `agb.model` - Data model framework
- `agb.string` - Text encoding/decoding

**Data Model Framework:**
The `agb.model` module provides a type system that mirrors C data structures for working with binary Game Boy Advance data. The framework is built around a base [`Type`](src/agb/model/type.py) class that defines the interface for all data types:

**Core Type Hierarchy:**
- **[`ScalarType`](src/agb/model/scalar_type.py)** - Basic data types like `u8`, `u16`, `u32`, `s8`, `s16`, `s32`, `pointer`
  - Can be associated with constant tables for named values
  - Handles endianness and binary packing automatically
- **[`Structure`](src/agb/model/structure.py)** - C-style structs with named fields
  - Supports complex layouts with priorities and dependencies
- **[`ArrayType`](src/agb/model/array.py)** - Fixed and variable-size arrays
  - [`FixedSizeArrayType`](src/agb/model/array.py) - Arrays with compile-time known size
  - [`VariableSizeArrayType`](src/agb/model/array.py) - Arrays whose size depends on other fields
- **[`UnboundedArrayType`](src/agb/model/unbounded_array.py)** - Null-terminated arrays
- **[`PointerType`](src/agb/model/pointer.py)** - Typed pointers to other data structures
- **[`BitfieldType`](src/agb/model/bitfield.py)** - Packed bit fields within scalars
- **[`UnionType`](src/agb/model/union.py)** - C-style unions
- **[`StringType`](src/agb/model/string.py)** - Game text with encoding support

**Key Operations:**
All types support these core operations:
- `from_data()` - Parse binary data from ROM into Python objects
- `to_assembly()` - Generate assembly code from Python objects
- `size()` - Calculate memory footprint
- `get_constants()` - Extract required constant definitions

Example usage:
```python
from agb.model.scalar_type import ScalarType
from agb.model.structure import Structure

# Define a scalar associated with constants
map_type = ScalarType('u8', constant='map_types')

# Define a C-style struct
class MapHeader(Structure):
    def __init__(self):
        super().__init__([
            ('width', 'u32', 0),
            ('height', 'u32', 0),
            ('tileset_primary', 'pointer', 0),
            ('tileset_secondary', 'pointer', 0),
        ])

# Parse from ROM data
header = map_header_type.from_data(rom, 0x8000000, project, [], [])
print(f"Map size: {header['width']}x{header['height']}")
```

### 🗺️ pymap - Map Editor & Project Manager
A full-featured map editor and project management system built on pyagb.

**Features:**
- **Visual Map Editor**: Graphical interface for editing game maps
- **Tileset Management**: Import, edit, and organize tilesets
- **Event Editing**: Place and configure NPCs, warps, triggers, and signposts
- **Connection System**: Link maps together with seamless transitions
- **Smart Shapes**: Advanced block placement tools
- **Project Organization**: Manage multiple maps, tilesets, and assets
- **Undo/Redo Support**: Full history tracking for all edits

**Key Components:**
- **GUI Application**: Qt-based visual editor
- **Project System**: JSON-based project files with asset management
- **Data Models**: Structured definitions for headers, footers, events
- **Rendering Engine**: Real-time map visualization
- **Export Tools**: Generate assembly code from project data

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pyagb

# Install in development mode
pip install -e .
```

## Quick Start

### Using pymap GUI
```bash
# Launch the map editor
python -m pymap.gui
```

### Using pyagb programmatically
```python
import agb.image
import agb.palette

# Load a GBA image
image, palette = agb.image.from_file('tileset.png')
```

### Project Management
```python
from pymap.project import Project

# Load an existing project
project = Project('my_hack.pmp')

# Load a map header
header, label, namespace = project.load_header('0', '0')

# Load associated footer (map data)
footer, footer_idx, smart_shapes = project.load_footer(label)
```

## Command Line Tools

### pymap2s - Compile project data to assembly
```bash
python -m pymap pymap2s project.pmp -o output.s --project
```

### bin2s - Convert binary files to assembly
```bash
python -m pymap bin2s input.bin -o output.s
```

### pypreproc - Preprocess assembly/C files
```bash
python -m pymap pypreproc input.s project.pmp -o output.s
```

## Project Structure

```
├── src/
│   ├── agb/              # Core AGB library
│   │   ├── types.py      # GBA data type definitions
│   │   ├── image.py      # Image processing
│   │   ├── palette.py    # Palette management
│   │   └── model/        # Data model framework
│   └── pymap/            # Map editor and project tools
│       ├── gui/          # Qt-based GUI components
│       ├── project.py    # Project management
│       ├── model/        # Map data structures
│       └── compile.py    # Code generation
├── pyproject.toml        # Project configuration
└── README.md
```

## Configuration

Projects use JSON configuration files that define:
- Data type mappings for different game versions
- File paths and organization
- Graphics formats and constraints
- Event type definitions
- Assembly generation settings

Example project structure:
```
my_hack/
├── my_hack.pmp           # Main project file
├── my_hack.pmp.config    # Configuration
├── my_hack.pmp.constants # Constants definitions
```

## Supported Features

### Map Editing
- ✅ Visual block-based map editing
- ✅ Multiple layer support
- ✅ Border and connection management
- ✅ Real-time preview
- ✅ Smart shape tools for efficient editing

### Asset Management
- ✅ PNG import/export for graphics
- ✅ Automatic palette optimization
- ✅ Tileset organization and reuse
- ✅ Animation support

### Data Export
- ✅ Assembly code generation
- ✅ C header generation
- ✅ Binary data export
- ✅ Project compilation

### Game Compatibility
Primarily tested with Pokemon games (Ruby/Sapphire/Emerald/FireRed/LeafGreen) but designed to be extensible for other GBA titles.

## Contributing

This project welcomes contributions! Areas of interest:
- GUI improvements
- Performance optimizations
- Documentation
- Bug fixes

## License

See [`LICENSE`](LICENSE) file for details.
