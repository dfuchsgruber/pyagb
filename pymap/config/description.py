# File that contains descriptions for different configurations

config_descriptions = {
    "macro" : " Define the configuration for constant library",
    "macro.as" : "Define the configuration for exporting the constants into assembly format"
    "macro.as.path" : "Define a file path to which assembly macros will be exported. {0} will be replaced with the label of the constant table",
    "macro.as.directive" : "Define which directive is used when an assembly file includes constants. {0} will be replaced with the label of the constant table",
    "macro.as.include" : "Define a set of constants that will exported into into assembly header files. Each constant table is identified by its label",
    "c.as" : "Define the configuration for exporting the constants into c format",
    "c.as.path" : "Define a file path to which c macros will be exported. {0} will be replaced with the label of the constant table",
    "c.as.directive" : "Define which directive is used when an c file includes constants. {0} will be replaced with the label of the constant table",
    "c.as.include" : "Define a set of constants that will exported into into c header files. Each constant table is identified by its label",
    "pymapex" : "Define configuration for the pymap map exporter",
    "pymapex.backend" : "Define the backend invokes for the pymap map exporter (scripts, tileset...)",
    "pymapex.backend.ow_script" : "Define an os command that is invoked when a script offset is encountered. {0} will be replaced with the path to the rom file. {1} will be replaced with the path to the project file. {2} will be replaced with the offset (in hex) of the script (e.g. 0x800000). {3} will be replaced with the directory of the exporter's output. {4} will contain a context dependent token: lscr, person, trigger or sign",
    "pymapex.backend.tileset" : "Define an os command that is invoked when a maptileset offset is encountered. {0} will be replacd wiht the path to the rom file. {1} will be replaced with the path to the project file. {2} will be replaced with the offset (in hex) of the tileset (e.g. 0x800000). {3} will be replaced with the tilesets index relative to the tileset table defined in pymapex's configuration. {4} will be replaced with the directory of the exporter's output.",
    "pymapex.batch" : "Define the configuration for the pymap exporter's batch export system",
    "pymapex.batch.map_output_path" : "Defines the filepath (without the .pmh extension) of a mapheader. Directories will be created recursively. {0} will be replaced with the map bank. {1} will be replaced with the map id in the bank. {2} will be replaced with the offset of the map.",
    "pymapex.batch.map_symbol" : "Defines the base symbol a mapheader will be associated with. For e.g. the symbol foo substructure elements such as the footer will be called foo_footer. {0} will be replaced with the map bank. {1} will be replacd with the map id in the bank. {2] will be replaced with the offset of the map.",
    "pymapex.maptable" : "Defines the maptable (array of pointers to mapbanks) the exporter uses.",
    "pymapex.constants" : "Defines all constants a map will use (note that the constants a script on a map uses might include way more tables which do not have to inclueded here). Those constant tables are used to replace values during the exporting with their respective counterparts."
    "pysetex" : "Define the configuration for the pymap tileset exporter.",
    "pysetex.tileset_base" : "Defines the base offset tilesets are stored at (i.e. the offset of Advance Map's Tileset0).",
    "pysetex.default_symbol" : "Defines the default symbol a tileset will associated with when no other information is specified during exporting. {0} will be replaced by the tileset's offset (in hex), e.g. 0x800000. {1} will be replaced with the tileset's number repsective to the tileset base, i.e. the offest of Advance Map's Tileset0.",
    "pymap2s" : "Define the configuration for the pymap mapheader compiler.",
    "pymap2s.constants" : "Define all constant tables that the generated assembly files should include using the specified directives. A constant table is refered to by its label",
    "pybuild" : "Define the configuration for the automatic building system",
    "pybuild.commands" : "Define the commands and their flags required for the system. You have to have armips and devkitPro installed.",
    "pybuild.commands.as" : "Define the settings for arm-none-eabi-as, the assembler",
    "pybuild.command.as.command" : "Name the command "