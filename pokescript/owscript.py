
from agb import agbrom
from pymap import project
from pokestring import pstring
import sys
import getopt
import os

TYPE_SCRIPT = 0
TYPE_MOVEMENT = 1
TYPE_STRING = 2
TYPE_MART = 3


LIBFILE = "lib/owscript.pml"
verbose = True
tablefile = "../table.tbl"

def _mkdirs(dir):
    if not os.path.exists(dir):
        print("Creating directory {0}".format(dir))
        os.makedirs(dir)


class Command:
    """ Class to store a command """
    def __init__(self, name, params, callback=lambda tree, rom, offset : None, ends_section = False):
        self.name = name
        self.params = params
        self.callback = callback
        self.ends_section = ends_section
        self.macro_export = None

    def size(self, rom, offset):
        size = 1
        for param in self.params:
            size += param.type.size
        return size

    def export(self, offset, used_constants):
        tokens = [self.name]
        offset += 1
        for param in self.params:
            tokens.append(param.type.export(offset, used_constants))
            offset += param.type.size
        return " ".join(tokens)

    def get_ends_section(self, rom, offset): return self.ends_section

    def to_macro(self, id):
        """ Exports the command as assembly macro"""
        macro = ".macro "+self.name+" "
        macro += " ".join(param.name for param in self.params) + "\n"
        macro += ".byte "+hex(id)+"\n"
        macro += "\n".join(param.to_macro() for param in self.params) + "\n"
        return macro + ".endm"

class Param:
    """ Class to hold params """
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def to_macro(self):
        """ Exports the param as asssembly macro"""
        if self.type.size == 4:
            return ".word \\{0}".format(self.name)
        if self.type.size == 2:
            return ".hword \\{0}".format(self.name)
        if self.type.size == 1:
            return ".byte \\{0}".format(self.name)
        raise Exception("Can not export macro for parameter {0} of size {1}.".format(self.name, str(self.type.size)))


class ParamType:
    """ Class to hold a param type
    export func is a function offset, used_constants_set -> string to consume data and retrieve an assembly string """
    def __init__(self, size, export_func):
        self.size = size
        self.export_func = export_func

    def export(self, offset, used_constants):
        return self.export_func(offset, used_constants)

class Owscript_Exploration_tree:
    """ Class to explore a script offset """

    def __init__(self, rom, proj):
        self.rom = rom
        self.offsets = []
        self.assemblies = {
            TYPE_SCRIPT : [],
            TYPE_MART : [],
            TYPE_MOVEMENT : [],
            TYPE_STRING : []
        }
        self.pedantic = False

        # Make the command table
        self.make_commands()

        self.project = proj

        # A none project is only supported
        # for macro exporting and under
        # no circumstances save for exploring
        # offsets
        if self.project is not None:
            # Read the config file
            with open(self.project.path + ".config", "r") as f:
                self.config = eval(f.read())["owscript_export"]

            # Initialize the explored offset
            self.load_exported_offsets()
        
    
    def explore(self, offset, verbose=verbose, pedantic=False):
        """ Explores an offset of a script tree """
        self.offsets.append(offset)
        self.pedantic = pedantic

        # Explore all offsets in self.offsets
        while len(self.offsets):
            offset = self.offsets.pop()
            if offset not in self.explored_offsets:
                if verbose: print("Exploring offset {0}".format(hex(offset)))
                label = self.config["script_label"].format(hex(offset))
                
                # Keep up a set for all constants used
                used_constants = set()

                # Add this offset to the explored offsets
                self.explored_offsets[offset] = TYPE_SCRIPT #We worked through this offset

                # Create the assembly and iterate through all commands
                assembly = ".global {0}\n\n{0}:\n".format(label)
                if offset < 0: 
                    print("Warning. Resolving something that is not a pointer and got offset {0} < 0.".format(
                        hex(offset)
                    ))
                else:
                    while True:
                        # Try retrieving the command id
                        try:
                            cmd_id = self.rom.u8(offset)
                        except Exception as ex:
                            print("Error at exporting script {0}, offset of error {1}. Rom \
                            not hold offset".format(label, hex(offset)))
                            raise ex
                        # Try retrieving the command
                        try:
                            cmd = self.owscript_cmds[cmd_id]
                        except Exception as ex:
                            print("Error at exporting script {0}, offset of error {1}. \
                            Undefined command {2}".format(label, hex(offset), hex(cmd_id)))
                            raise ex
                        # Export command structure
                        try:
                            assembly += cmd.export(offset, used_constants) + "\n"
                        except Exception as ex:
                            print("Error at exporting script {0}, offset of error {1}. \
                            Could not export command {2}".format(label, hex(offset), hex(cmd_id)))
                            raise ex
                        # Break the iteration of the command ends a (sub)script
                        if cmd.get_ends_section(self.rom, offset):
                            break
                        offset += cmd.size(self.rom, offset)
                    
                    self.assemblies[TYPE_SCRIPT].append((label, assembly, used_constants))
                

    def make_commands(self):
        """ Makes the command list """

        # Define Parameter types, can be extended 
        # and stores callbacks to exploration funcs 
        # that yield the exported label
        # and also proceed exporting (recursively)

        BYTE = ParamType(1, self.explore_functor(1))
        HWORD = ParamType(2, self.explore_functor(2))
        WORD =  ParamType(4, self.explore_functor(4))
        SCRIPT_REFERENCE = ParamType(4, self.explore_script_reference)
        ITEM = ParamType(2, self.constant_explore_functor(2, "items", var_access = True))
        FLAG = ParamType(2, self.constant_explore_functor(2, "flags", var_access = True))
        MOVEMENT_LIST = ParamType(4, self.explore_movement_list)
        STRING = ParamType(4, self.explore_string_reference)
        POKEMON = ParamType(2, self.constant_explore_functor(2, "species", var_access = True))
        ATTACK = ParamType(2, self.constant_explore_functor(2, "attacks", var_access = True))
        MART = ParamType(4, self.explore_mart_list)
        VAR = ParamType(2, self.constant_explore_functor(2, "vars"))
        ORD = ParamType(1, self.constant_explore_functor(1, "ordinals"))
        STD = ParamType(1, self.constant_explore_functor(1, "callstds"))
        MUSIC = ParamType(2, self.constant_explore_functor(2, "songs"))
        MAP_WEATHER = ParamType(1, self.constant_explore_functor(1, "map_weathers"))
        BEHAVIOUR = ParamType(1, self.constant_explore_functor(1, "person_behaviours"))

        # Make rom accessible for the factory such that
        # trainerbattles can be created from an offset.
        global_rom = self.rom

        # Create a factory class for trainerbattle
        class CommandTrainerbattleFactory:
            """ Explicit class for trainerbattle cmd """
            def __init__(self):
                pass
            
            def make_command_by_type(self, type):
                if type in (0,5):
                    return Command("trainerbattlestd", [Param("kind", BYTE), Param("trainer", HWORD),
                    Param("people", HWORD), Param("str_challange", STRING), Param("str_defeat", STRING)])
                elif type in (1,2):
                    return Command("trainerbattlecont", [Param("kind", BYTE), Param("trainer", HWORD),
                    Param("people", HWORD), Param("str_challange", STRING), Param("str_defeat", STRING),
                    Param("continuation", SCRIPT_REFERENCE)], ends_section = True)
                elif type in (4,7):
                    return Command("trainerbattledouble", [Param("kind", BYTE), Param("trainer", HWORD),
                    Param("people", HWORD), Param("str_challange", STRING), Param("str_defeat", STRING),
                    Param("continuation_one_poke", SCRIPT_REFERENCE)])
                elif type == 3:
                    return Command("trainerbattlenotinterrupting", [Param("kind", BYTE), Param("trainer", HWORD),
                    Param("people", HWORD), Param("str_defeat", STRING)])
                elif type == 9:
                    return Command("trainerbattlelosable", [Param("kind", BYTE), Param("trainer", HWORD),
                    Param("people", HWORD), Param("str_defeat", STRING), Param("str_loss", STRING)])
                elif type in (6,8):
                    return Command("trainerbattle6", [Param("kind", BYTE), Param("trainer", HWORD),
                    Param("people", HWORD), Param("str_challange", STRING), Param("str_defeat", STRING),
                    Param("str_var1", WORD), Param("continuation", SCRIPT_REFERENCE)])

            def get_ends_section(self, rom, offset): return self.make_command(rom, offset).ends_section

            def make_command(self, rom, offset):
                type = rom.u8(offset + 1)
                return self.make_command_by_type(type)
                
            def size(self, rom, offset):
                return self.make_command(rom, offset).size(rom, offset)

            def export(self, offset, used_constants):
                return self.make_command(global_rom, offset).export(offset, used_constants)

            def to_macro(self, id):
                """ Exports all types as macro """
                return "\n\n".join(self.make_command_by_type(type).to_macro(id) for type in {0, 1, 4, 3, 9, 6})

        self.owscript_cmds = [
            #0x00
            Command("nop", []),
            Command("nop1", []),
            Command("end", [], ends_section = True),
            Command("return", [], ends_section = True),
            Command("call", [Param("subscript", SCRIPT_REFERENCE)]),
            Command("goto", [Param("script", SCRIPT_REFERENCE)], ends_section = True),
            Command("gotoif", [Param("condition", ORD), Param("subscript", SCRIPT_REFERENCE)]),
            Command("callif", [Param("condition", ORD), Param("script", SCRIPT_REFERENCE)]),
            #0x08
            Command("gotostd", [Param("std", STD)], ends_section = True),
            Command("callstd", [Param("std", STD)]),
            Command("gotostdif", [Param("condition", BYTE), Param("std", STD)]),
            Command("callstdif", [Param("condition", BYTE), Param("std", STD)]),
            Command("jumpram", []),
            Command("killscript", []),
            Command("set_x203AA3C", [Param("value", BYTE)]),
            Command("loadpointer", [Param("bank", BYTE), Param("pointer", STRING)]),
            #0x10
            Command("set_intern_bank", [Param("bank", BYTE), Param("value", BYTE)]),
            Command("writebytetooffset", [Param("value", BYTE), Param("offset", WORD)]),
            Command("loadbytefrompointer", [Param("bank", BYTE), Param("offset", WORD)]),
            Command("setbyte", [Param("bank", BYTE), Param("offset", WORD)]),
            Command("copyscriptbanks", [Param("dst_bank", BYTE), Param("src_bank", BYTE)]),
            Command("copybyte", [Param("dst", WORD), Param("src", WORD)]),
            Command("setvar", [Param("var", VAR), Param("value", HWORD)]),
            Command("addvar", [Param("var", VAR), Param("value", HWORD)]),
            #0x18
            Command("subvar", [Param("var", VAR), Param("value", HWORD)]),
            Command("copyvar", [Param("dst", VAR), Param("src", VAR)]),
            Command("copyvarifnotzero", [Param("dst", VAR), Param("src", ITEM)]),
            Command("comparebanks", [Param("bank1", HWORD), Param("bank2", HWORD)]),
            Command("comparebanktobyte", [Param("bank", BYTE), Param("value", BYTE)]),
            Command("comparebanktofarbyte", [Param("bank", BYTE), Param("offset", WORD)]),
            Command("comparefarbytetobank", [Param("offset", WORD), Param("bank", BYTE)]),
            Command("comparefarbytetobyte", [Param("offset", WORD), Param("value", BYTE)]),
            #0x20
            Command("comparefarbytes", [Param("offset1", WORD), Param("offset2", WORD)]),
            Command("compare", [Param("var", VAR), Param("value", HWORD)]),
            Command("comparevars", [Param("var1", VAR), Param("var2", VAR)]),
            Command("callasm", [Param("function", WORD)]),
            Command("callasmandwaitstate", [Param("function", WORD)]),
            Command("special", [Param("special_id", HWORD)]),
            Command("special2", [Param("varresult", HWORD), Param("speical_id", HWORD)]),
            Command("waitstate", []),
            #0x28
            Command("pause", [Param("frames", HWORD)]),
            Command("setflag", [Param("flag", FLAG)]),
            Command("clearflag", [Param("flag", FLAG)]),
            Command("checkflag", [Param("flag", FLAG)]),
            Command("cmd2c", []),
            Command("cmd2d", []),
            Command("resetvolatilevars", []),
            Command("sound", [Param("sound", HWORD)]),
            #0x30
            Command("checksound", []),
            Command("fanfare", [Param("fanfare", HWORD)]),
            Command("waitfanfare", []),
            Command("playsong", [Param("song", MUSIC), Param("feature", BYTE)]),
            Command("playsong2", [Param("song", MUSIC)]),
            Command("songfadedefault", []),
            Command("fadesong", [Param("song", MUSIC)]),
            Command("fadeout", [Param("speed", BYTE)]),
            #0x38
            Command("fadein", [Param("speed", BYTE)]),
            Command("warp", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            Command("warpmuted", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            Command("warpwalk", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            Command("warphole", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            Command("warpteleport", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            Command("warp3", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            Command("setwarpplace", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            #0x40
            Command("warp4", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            Command("warp5", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            Command("getplayerpos", [Param("varx", HWORD), Param("vary", HWORD)]),
            Command("countpokemon", []),
            Command("additem", [Param("item", ITEM), Param("quantity", HWORD)]),
            Command("removeitem", [Param("item", ITEM), Param("quantity", HWORD)]),
            Command("checkitemroom", [Param("item", ITEM), Param("quantity", HWORD)]),
            Command("checkitem", [Param("item", ITEM), Param("quantity", HWORD)]),
            #0x48
            Command("checkitemtype", [Param("item", ITEM)]),
            Command("addpcitem", [Param("item", ITEM), Param("quantity", HWORD)]),
            Command("checkpcitem", [Param("item", ITEM), Param("quantity", HWORD)]),
            Command("cmd4b", [Param("unused", HWORD)]),
            Command("cmd4c", [Param("unused", HWORD)]),
            Command("cmd4d", [Param("unused", HWORD)]),
            Command("cmd4e", [Param("unused", HWORD)]),
            Command("applymovement", [Param("people", HWORD), Param("movement", MOVEMENT_LIST)]),
            #0x50
            Command("applymovementonmap", [Param("people", HWORD), Param("movement", MOVEMENT_LIST), Param("bank", BYTE), Param("map", BYTE)]),
            Command("waitmovement", [Param("num_people", HWORD)]),
            Command("waitmovementonmap", [Param("num_people", HWORD), Param("bank", BYTE), Param("map", BYTE)]),
            Command("hidesprite", [Param("people", HWORD)]),
            Command("hidespriteonmap", [Param("people", HWORD), Param("bank", BYTE), Param("map", BYTE)]),
            Command("showsprite", [Param("people", HWORD)]),
            Command("showspriteonmap", [Param("people", HWORD), Param("bank", BYTE), Param("map", BYTE)]),
            Command("movesprite", [Param("people", HWORD), Param("x", HWORD), Param("y", HWORD)]),
            #0x58
            Command("spritevisible", [Param("people", HWORD), Param("bank", BYTE), Param("map", BYTE)]),
            Command("spriteinvisible", [Param("people", HWORD), Param("bank", BYTE), Param("map", BYTE)]),
            Command("faceplayer", []),
            Command("spriteface", [Param("people", HWORD), Param("facing", BYTE)]),
            CommandTrainerbattleFactory(),
            Command("repeattrainerbattle",[]),
            Command("endtrainerbattle",[]),
            Command("endtrainerbattle2",[]),
            #0x60
            Command("checktrainerflag", [Param("id", HWORD)]),
            Command("cleartrainerflag", [Param("id", HWORD)]),
            Command("settrainerflag", [Param("id", HWORD)]),
            Command("movesprite2", [Param("people", HWORD), Param("x", HWORD), Param("y", HWORD)]),
            Command("moveoffscreen", [Param("people", HWORD)]),
            Command("spritebehave", [Param("people", HWORD), Param("behaviour", BEHAVIOUR)]),
            Command("waitmsg", []),
            Command("preparemsg", [Param("str", STRING)]),
            #0x68
            Command("closeonkeypress", []),
            Command("lockall", []),
            Command("lock", []),
            Command("releaseall", []),
            Command("release", []),
            Command("waitkeypress", []),
            Command("yesnobox", [Param("tilex", BYTE), Param("tiley", BYTE)]),
            Command("multichoice", [Param("tilex", BYTE), Param("tiley", BYTE), Param("id", BYTE), Param("not_escapable", BYTE)]),
            #0x70
            Command("multichoice2", [Param("tilex", BYTE), Param("tiley", BYTE), Param("id", BYTE), Param("default", BYTE), Param("not_escapable", BYTE)]),
            Command("multichoice3", [Param("tilex", BYTE), Param("tiley", BYTE), Param("id", BYTE), Param("num_per_row", BYTE), Param("not_escapable", BYTE)]),
            Command("showbox", [Param("tilex", BYTE), Param("tiley", BYTE), Param("tilew", BYTE), Param("tileh", BYTE)]),
            Command("hidebox", [Param("tilex", BYTE), Param("tiley", BYTE), Param("tilew", BYTE), Param("tileh", BYTE)]),
            Command("clearbox", [Param("tilex", BYTE), Param("tiley", BYTE), Param("tilew", BYTE), Param("tileh", BYTE)]),
            Command("showpokepic", [Param("species", POKEMON), Param("tilex", BYTE), Param("tiley", BYTE)]),
            Command("hidepokepic", []),
            Command("cmd77", []),
            #0x78
            Command("braille", [Param("brialledata", STRING)]),
            Command("givepokemon", [Param("species", POKEMON), Param("level", BYTE), Param("item", ITEM),
            Param("filler1", WORD), Param("filler2", WORD), Param("filler3", BYTE)]),
            Command("giveegg", [Param("species", POKEMON)]),
            Command("setpokemonpp", [Param("teamslot", BYTE), Param("attackslot", BYTE), Param("pp", HWORD)]),
            Command("checkattack", [Param("move", ATTACK)]),
            Command("bufferpokemon", [Param("buffer", BYTE), Param("species", POKEMON)]),
            Command("bufferfirstpokemon", [Param("buffer", BYTE)]),
            Command("bufferpartypokemon", [Param("buffer", BYTE), Param("teamsplot", HWORD)]),
            #0x80
            Command("bufferitem", [Param("buffer", BYTE), Param("item", ITEM)]),
            Command("cmd81", [Param("buffer", BYTE), Param("deco", HWORD)]),
            Command("bufferattack", [Param("buffer", BYTE), Param("move", ATTACK)]),
            Command("buffernumber", [Param("buffer", BYTE), Param("var", HWORD)]),
            Command("bufferstd", [Param("buffer", BYTE), Param("std", HWORD)]),
            Command("bufferstring", [Param("buffer", BYTE), Param("str", STRING)]),
            Command("pokemart", [Param("mart", MART)]),
            Command("pokemart2", [Param("mart", MART)]),
            #0x88
            Command("pokemart3", [Param("mart", MART)]),
            Command("pokecasino", [Param("var", HWORD)]),
            Command("cmd8a", [Param("param1", BYTE), Param("param2", BYTE)], Param("param3", BYTE)),
            Command("cmd8b", []),
            Command("cmd8c", []),
            Command("cmd8d", []),
            Command("cmd8e", []),
            Command("random", [Param("module", HWORD)]),
            #0x90
            Command("givemoney", [Param("amount", WORD), Param("execbank", BYTE)]),
            Command("paymoney", [Param("amount", WORD), Param("execbank", BYTE)]),
            Command("checkmoney", [Param("amount", WORD), Param("execbank", BYTE)]),
            Command("showmoney", [Param("tilex", BYTE), Param("tiley", BYTE), Param("execbank", BYTE)]),
            Command("hidemoney", [Param("tilex", BYTE), Param("tiley", BYTE)]),
            Command("updatemoney", [Param("tilex", BYTE), Param("tiley", BYTE), Param("execbank", BYTE)]),
            Command("cmd96", [Param("param1", HWORD)]),
            Command("fadescreen", [Param("effect", BYTE)]),
            #0x98
            Command("fadescreenspeed", [Param("effect", BYTE), Param("speed", BYTE)]),
            Command("darken", [Param("flashradius", HWORD)]),
            Command("lighten", [Param("flashradius", HWORD)]),
            Command("preparemsg2", [Param("str", STRING)]),
            Command("doanimation", [Param("animid", BYTE)]),
            Command("setanimation", [Param("animid", BYTE), Param("var", HWORD)]),
            Command("checkanimation", [Param("animid", HWORD)]),
            Command("sethealingplace", [Param("id", HWORD)]),
            #0xA0
            Command("checkgender", []),
            Command("cry", [Param("species", POKEMON), Param("effect", HWORD)]),
            Command("setmaptile", [Param("x", HWORD), Param("y", HWORD), Param("block", HWORD), Param("attribute", HWORD)]),
            Command("resetweather", []),
            Command("setweather", [Param("weather", MAP_WEATHER)]),
            Command("doweather", []),
            Command("cmda6", [Param("param1", BYTE)]),
            Command("setmapfooter", [Param("footer", HWORD)]),
            #0xA8
            Command("spritelevelup", [Param("people", HWORD), Param("bank", BYTE), Param("map", BYTE), Param("oamfield43", BYTE)]),
            Command("restorespritelevel", [Param("people", HWORD), Param("bank", BYTE), Param("map", BYTE)]),
            Command("createsprite", [Param("picture", BYTE), Param("id", BYTE), Param("x", HWORD), Param("y", HWORD),
            Param("behaviour", BEHAVIOUR), Param("behaviour_range", BYTE)]),
            Command("spriteface2", [Param("people", BYTE), Param("facing", BYTE)]),
            Command("setdooropened", [Param("x", HWORD), Param("y", HWORD)]),
            Command("setdoorclosed", [Param("x", HWORD), Param("y", HWORD)]),
            Command("doorchange", []),
            Command("setdooropened2", [Param("x", HWORD), Param("y", HWORD)]),
            #0xB0
            Command("setdoorclosed2", [Param("x", HWORD), Param("y", HWORD)]),
            Command("nop2", []),
            Command("nop3", []),
            Command("getcoins", [Param("var", HWORD)]),
            Command("givecoins", [Param("amount", HWORD)]),
            Command("removecoins", [Param("amount", HWORD)]),
            Command("setwildbattle", [Param("species", POKEMON), Param("level", BYTE), Param("item", ITEM)]),
            Command("dowildbattle", []),
            #0xB8
            Command("setvirtualscriptdisplacement", [Param("offset", WORD)]),
            Command("virtualgoto", [Param("scriptplusdispl", WORD)]),
            Command("virtualcall", [Param("scriptplusdispl", WORD)]),
            Command("virtualgotoif", [Param("condition", BYTE), Param("scriptplusdispl", WORD)]),
            Command("virutalcallif", [Param("condition", BYTE), Param("scriptplusdispl", WORD)]),
            Command("virtualmsgbox", [Param("str", STRING)]),
            Command("virtualloadpointer", [Param("str", STRING)]),
            Command("virtualbuffer", [Param("buffer", BYTE), Param("str", STRING)]),
            #0xC0
            Command("showcoins", [Param("tilex", BYTE), Param("tiley", BYTE)]),
            Command("hidecoins", [Param("tilex", BYTE), Param("tiley", BYTE)]),
            Command("updatecoins", [Param("tilex", BYTE), Param("tiley", BYTE)]),
            Command("savincrementkey", [Param("key", BYTE)]),
            Command("warp6", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            Command("waitcry", []),
            Command("bufferboxname", [Param("buffer", BYTE), Param("box", HWORD)]),
            Command("textcolor", [Param("color", BYTE)]),
            #0xC8
            Command("menucreate", [Param("menu", WORD)]),
            Command("menuflush", []),
            Command("signmsg", []),
            Command("normalmsg", []),
            Command("savcomparekey", [Param("key", BYTE), Param("value", WORD)]),
            Command("setobedience", [Param("teamslot", HWORD)]),
            Command("checkobedience", [Param("teamslot", HWORD)]),
            Command("executeram", []),
            #0xD0
            Command("setworldmapflag", [Param("flag", FLAG)]),
            Command("warpteleport2", [Param("bank", BYTE), Param("map", BYTE), Param("exit", BYTE), Param("x", HWORD), Param("y", HWORD)]),
            Command("setcatchlocation", [Param("teamslot", HWORD), Param("namespace", BYTE)]),
            Command("braille2", [Param("brailledata", STRING)]),
            Command("bufferitems", [Param("buffer", BYTE), Param("item", ITEM), Param("quantity", HWORD)]),
            Command("singlemovement", [Param("target", HWORD), Param("move", HWORD)])
        ]

    def constant_explore_functor(self, size, label, var_access=False):
        """ Functor that returns an expolore
        function for any constant """
        def explore(offset, used_constants):
            # Read value by std access functor
            _label = label
            value = int(self.explore_functor(size)(offset, used_constants), 16)
            # Try to parse var first
            if var_access:
                for var_range in self.config["var_range"]:
                    if value in var_range:
                        _label = "vars"
            if _label in self.config["constants"]:
                used_constants.add(_label)
                return self.project.constants.constantize(value, label, pedantic=self.pedantic)
            else:
                return hex(value)
        return explore

    def explore_functor(self, size):
        """ Functor that returns an access function
        for a data type. """
        def explore(offset, used_constants):
            if size == 1:
                value = self.rom.u8(offset)
            elif size == 2:
                value = self.rom.u16(offset)
            elif size == 4:
                value = self.rom.u32(offset)
            elif size == -1:
                value = self.rom.s8(offset)
            elif size == -2:
                value = self.rom.s16(offset)
            elif size == -4:
                value = self.rom._int(offset)
            else:
                raise Exception("Could not make constant functor for size {0}".format(size))
            return hex(value)
        return explore

    def explore_script_reference(self, offset, used_constants):
        """ Explores a script reference """
        target = self.rom.pointer(offset)
        if target not in range(0, 0x2000000):
            raise Exception("Script offset {0} is not a valid target.".format(
                hex(target)
            ))
        self.offsets.append(target)
        label = self.config["script_label"].format(hex(target))
        return label
    
    def explore_movement_list(self, offset, used_constants):
        """ Explores an offset as reference to a movement list """

        offset = self.rom.pointer(offset)
        if offset not in range(0, 0x2000000):
            raise Exception("Movement list offset {0} is not contained in ROM.".format(
                hex(offset)
            ))
        label = self.config["movement_list_label"].format(hex(offset))
        
        # Explore if not explored
        if not offset in self.explored_offsets:
            assembly = ".global {0}\n\n{0}:\n".format(label)
            self.explored_offsets[offset] = TYPE_MOVEMENT
            explore = True
            while explore:
                next_move = self.rom.u8(offset)
                if next_move == 0xFE: # MOVE END
                    explore = False
                if "movements" in self.config["constants"]:
                    next_move = self.project.constants.constantize(next_move, "movements", pedantic=self.pedantic)
                    used_constants = set(["movements"])
                else:
                    next_move = hex(next_move)
                    used_constants = set()
                assembly += "\t.byte {0}\n".format(
                    next_move
                )
                offset += 1
            self.assemblies[TYPE_MOVEMENT].append((label, assembly, used_constants))
        return label
 
    def explore_mart_list(self, offset, used_constants):
        """ Explores an offset as reference to a mart list """

        offset = self.rom.pointer(offset)
        if offset not in range(0, 0x2000000):
            raise Exception("Mart list offset {0} is not contained in ROM.".format(
                hex(offset)
            ))
        label = self.config["mart_list_label"].format(hex(offset))

        # Explore if not explored
        if not offset in self.explored_offsets:
            assembly = ".global {0}\n\n.align 2\n{0}:\n".format(label)
            self.explored_offsets[offset] = TYPE_MART
            explore = True
            while explore:
                item = self.rom.u16(offset)
                if item == 0: 
                    explore = False 
                if "items" in self.config["constants"]:
                    item = self.project.constants.constantize(item, "items", pedantic=self.pedantic)
                    used_constants = set(["items"])
                else:
                    item = hex(item)
                    used_constants = set()
                assembly += ".hword {0}\n".format(
                    item
                )
                offset += 2       
            self.assemblies[TYPE_MART].append((label, assembly, used_constants))
        return label

    def explore_string_reference(self, offset, used_constants):
        """ Explores a string reference and exports the string by defined backend. """

        offset = self.rom.pointer(offset)
        if offset not in range(0, 0x2000000):
            raise Exception("String offset {0} is not contained in ROM.".format(
                hex(offset)
            ))

        str_backend_config = self.config["string_backend"]
        label = str_backend_config["label"].format(hex(offset))

        # Explore if not explored
        if not offset in self.explored_offsets:
            
            self.explored_offsets[offset] = TYPE_STRING
            if str_backend_config["use_string_backend"]:
                # Use the defined backend
                os.system(str_backend_config["exec"].format(hex(offset)))
                self.explored_offsets[offset] = TYPE_STRING
            else:
                # Use the intern pstring exporter
                intern_config = str_backend_config["intern"]

                ps = pstring.Pstring(intern_config["charmap"], terminator=intern_config["terminator"])
                string = ps.hex2str(self.rom, offset)
                assembly = ".global {0}\n\n{0}:\n.string {1} \"{2}\"\n".format(
                    label, intern_config["language"], string
                )
                self.assemblies[TYPE_STRING].append((label, assembly, set()))
        return label

    def export(self, directory):
        """ Exports a all assemblies to the corresponding directory. """
        for type in self.assemblies:
            for assembly_label, assembly, used_constants in self.assemblies[type]:
                # Create a preamble
                preamble = ""
                for label in used_constants:
                    preamble += self.project.constants.get_include_directive(label, "as") + "\n"
                
                preamble += "\n\n"

                # Output the file
                with open(os.path.join(directory, assembly_label + ".asm"), "w+") as f:
                    f.write(preamble + assembly)
        
    def export_singlefile(self, filepath):
        """ Exports all assemblies together as one file. """
        global_used_constants = set()
        assembled = ""

        for type in self.assemblies:
            for _, assembly, used_constants in self.assemblies[type]:
                for label in used_constants:
                    global_used_constants.update(used_constants)
                assembled += assembly + "\n\n"

        # Create the global preamble
        preamble = ""
        for label in global_used_constants:
            preamble += self.project.constants.get_include_directive(label, "as") + "\n"
        
        with open(filepath, "w+") as f:
            f.write(preamble + "\n\n" + assembled)

    def load_exported_offsets(self):
        """ Loads the exported offsets from the lib file
        specified in the configuration. """
        path = self.project.realpath(self.config["explored_offsets"])

        if not os.path.exists(path):
            # Create the lib file initially
            self.explored_offsets = {}
            with open(path, "w+") as f:
                f.write(str(self.explored_offsets))
        else:
            with open(path, "r") as f:
                self.explored_offsets = eval(f.read())


    def update_exported_offsets(self):
        """ Updates the exported offsets with all offsets
        from self.explored_offsets. """
        
        path = self.project.realpath(self.config["explored_offsets"])
        with open(path, "r") as f:
            self.explored_offsets.update(eval(f.read()))
        
        with open(path, "w+") as f:
            f.write(str(self.explored_offsets))


def main(argv):
    """ Shell method for this tool """
    try:
        opts, args = getopt.getopt(argv, "hs:o:", ["help", "verbose", "pedantic", "macro", "singlefile", "mkdirs"])
    except getopt.GetoptError:
        sys.exit(2)

    offset = None
    outdir = None
    verbose = False
    pedantic = False
    singlefile = False
    mkdirs = False

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("""Usage: owscriptex.py [opts] romfile projfile
Options:
    -o {dir}        :   Specify the output directory. Files will be named like the labels
                        of the contained assembly (e.g. dir/{label}.asm)
    -s {offset}     :   Define the offset to export
    --verbose       :   If used the exporter will be verbose. [default=False]
    --pedantic      :   If used the exporter will be pedantic when resolving values
                        and associating them with constants and thus raise Exceptions
                        when a value could not be resolved. [default=False]
    --singlefile    :   Exports all assemblies that were collected by recursively
                        exploring the given offset in one file rather than a file
                        for every assembly with a global label. The file will be
                        name after the script label of the root offset. [default=False]
    --mkdirs        :   Creates directories if neccessary. [default=False]
Alternative usage: owscriptex.py --macro macrofile
This will export an assembly header that contains all script command macros.
Macrofile specifies the filepath.""")
            sys.exit(0)
        if opt == "-s":
            offset = int(arg, 0)
        elif opt == "-o":
            outdir = arg
        elif opt == "--verbose":
            verbose = True
        elif opt == "--pedantic":
            pedantic = True
        elif opt == "--macro":
            # Special usage, export only macro file
            export_macro(args[0])
            exit(0)
        elif opt == "--singlefile":
            singlefile = True
        elif opt == "--mkdirs":
            mkdirs = True

    try:
        rom = agbrom.Agbrom(args[0])
    except:
        print("No rom file specified (see --help).")
        exit(1)
    
    try:
        proj = project.Project.load_project(args[1])
    except:
        print("No project file specified (see --help).")
        exit(1)
    
    if offset is None:
        print("No offset specifed (see --help)")
        exit(1)

    if outdir is None:
        print("No output directory specified (see --help)")
        exit(1)
    
    # Start exporting
    tree = Owscript_Exploration_tree(rom, proj)
    tree.explore(offset, verbose=verbose, pedantic=pedantic)

    # Update exported_offsets lib and export assemblies
    #tree.update_exported_offsets()

    _mkdirs(outdir)

    if singlefile:
        rootlabel = tree.config["script_label"].format(hex(offset))
        outfile = os.path.join(outdir, rootlabel + ".asm")
        tree.export_singlefile(outfile)
    else:
        tree.export(outdir)

def export_macro(path):
    """ Exports the ow script macros to a path. """
    # Create an empty tree
    tree = Owscript_Exploration_tree(None, None)
    macros = ""
    for i in range(len(tree.owscript_cmds)):
        macros += tree.owscript_cmds[i].to_macro(i) + "\n\n"

    with open(path, "w+") as f:
        f.write(macros)
    print("Sucessfully exported macro header file to {0}".format(path))

