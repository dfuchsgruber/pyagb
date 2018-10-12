import agb.types
from . import backend

class EventHeader:
    """ Class to model event headers. """
    
    def __init__(self):
        self.persons = []
        self.triggers = []
        self.warps = []
        self.signposts = []

    def from_data(self, rom, offset, project, context):
        """ Retrieves an event header and all events associated with it from rom data.
        
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
        
        Returns:
        --------
        self : EventHeader
            Self-reference.
        offset : int
            The offeset of the next bytes after the event header structure was processed
        """
        num_persons, num_warps, num_triggers, num_signposts = rom.u8[offset : offset + 4]
        persons_offset, warps_offset, triggers_offset, signposts_offset = rom.pointer[offset + 4 : offset + 20 : 4]

        # Retrieve persons
        self.persons = []
        for i in range(num_persons):
            self.persons.append(Person())
            _, persons_offset = self.persons[-1].from_data(rom, persons_offset, project, context + ['person', str(i)])

        # Retrieve warps
        self.warps = []
        for i in range(num_warps):
            self.warps.append(Warp())
            _, warps_offset = self.warps[-1].from_data(rom, warps_offset, project, context + ['warp', str(i)])

        # Retrieve triggers
        self.triggers = []
        for i in range(num_triggers):
            self.triggers.append(Trigger())
            _, triggers_offset = self.triggers[-1].from_data(rom, triggers_offset, project, context + ['trigger', str(i)])

        # Retrieve signposts
        self.signposts = []
        for i in range(num_signposts):
            # Use the signpost type to determine the signpost class
            self.signposts.append(Signpost())
            print(hex(signposts_offset))
            _, signposts_offset = self.signposts[-1].from_data(rom, signposts_offset, project, context + ['signpost', str(i)])

    def to_assembly(self):
        """ Creates an assembly representation of the event header.
        
        Returns:
        --------
        assembly : str
            The assembly representation of the event header.
        """
        event_to_assembly = lambda event: event.to_assembly()
        return '\n'.join([
            f'.byte {len(self.persons)}, {len(self.warps)}, {len(self.triggers)}, {len(self.signposts)}',
            '.word persons, warps, triggers, signposts',
            'persons:',
            '\n'.join(map(event_to_assembly, self.persons)),
            'warps:',
            '\n'.join(map(event_to_assembly, self.warps)),
            'triggers:',
            '\n'.join(map(event_to_assembly, self.triggers)),
            'signposts:',
            '\n'.join(map(event_to_assembly, self.signposts))
        ])

    def to_dict(self):
        """ Returns a json serializable representation of the event header.
        
        Returns:
        --------
        serializable : dict
            The dict representing this event header.
        """
        return {
            'persons' : [person.to_dict() for person in self.persons],
            'warps' : [warp.to_dict() for warp in self.warps],
            'triggers' : [trigger.to_dict() for trigger in self.triggers],
            'signposts' : [signpost.to_dict() for signpost in self.signposts]
        }
    
    def from_dict(self, serialized):
        """ Initializes the event header from a dictionary.
        
        Parameters:
        -----------
        serialized : dict
            Dict representing the event header.
        """
        for person_dict in serialized['persons']:
            self.persons.append(Person())
            self.persons[-1].from_dict(person_dict)
        for warp_dict in serialized['warps']:
            self.warps.append(Warp())
            self.warps[-1].from_dict(warp_dict)
        for trigger_dict in serialized['triggers']:
            self.triggers.append(Trigger())
            self.triggers[-1].from_dict(trigger_dict)
        for signpost_dict in serialized['signposts']:
            self.signposts.append(Signpost())
            self.signposts[-1].from_dict(signpost_dict)


class Event(agb.types.Structure):
    """ Superclass for map events """

    def __init__(self, structure):
        """ Constructor that passes arguments to the Structure-superclass
        constructor.
        
        Parameters:
        -----------
        structure : list of tuples (member, type, default)
            member : str
                The name of the member
            type : str
                The name of the type
            default : object
                The default initialization
        """
        super().__init__(structure)


class Person(Event):
    """ Class to model an person map event"""
    def __init__(self):
        super().__init__([
            ('target_index', agb.types.ScalarType('u8'), 0),
            ('picture', agb.types.ScalarType('u8'), 0),
            ('field_2', agb.types.ScalarType('u8'), 0),
            ('field_3', agb.types.ScalarType('u8'), 0),
            ('x', agb.types.ScalarType('s16'), 0),
            ('y', agb.types.ScalarType('s16'), 0),
            ('level', agb.types.ScalarType('u8'), 0),
            ('behaviour', agb.types.ScalarType('u8', constant='person_behaviours'), 0),
            ('behaviour_range', agb.types.ScalarType('u8'), 0),
            ('field_B', agb.types.ScalarType('u8'), 0),
            ('is_trainer', agb.types.ScalarType('u8'), 0),
            ('field_D', agb.types.ScalarType('u8'), 0),
            ('alter_radius', agb.types.ScalarType('u16'), 0),
            ('script', backend.OwScriptPointerType(), 0),
            ('flag', agb.types.ScalarType('u16', constant='flags'), 0),
            ('field_16', agb.types.ScalarType('u16'), 0)
        ])

class Trigger(Event):
    """ Class to model a trigger event """
    def __init__(self):
        super().__init__([
            ('x', agb.types.ScalarType('s16'), 0),
            ('y', agb.types.ScalarType('s16'), 0),
            ('level', agb.types.ScalarType('u8'), 0),
            ('field_5', agb.types.ScalarType('u8'), 0),
            ('var', agb.types.ScalarType('u16', constant='vars'), 0),
            ('value', agb.types.ScalarType('u16'), 0),
            ('field_A', agb.types.ScalarType('u8'), 0),
            ('field_B', agb.types.ScalarType('u8'), 0),
            ('script', backend.OwScriptPointerType(), 0)
        ])

class Warp(Event):
    """ Class to model a warp event """
    def __init__(self):
        super().__init__([
            ('x', agb.types.ScalarType('s16'), 0),
            ('y', agb.types.ScalarType('s16'), 0),
            ('level', agb.types.ScalarType('u8'), 0),
            ('target_idx', agb.types.ScalarType('u8'), 0),
            ('bank', agb.types.ScalarType('u8'), 0),
            ('map', agb.types.ScalarType('u8'), 0)
        ])

class Signpost(Event):
    """ Superclass to model signposts """
    def __init__(self):
        super().__init__([
            ('x', agb.types.ScalarType('s16'), 0),
            ('y', agb.types.ScalarType('s16'), 0),
            ('level', agb.types.ScalarType('u8'), 0),
            ('type', agb.types.ScalarType('u8'), 0),
            ('field_6', agb.types.ScalarType('u8'), 0),
            ('field_7', agb.types.ScalarType('u8'), 0),
            ('value', agb.types.UnionType({
                'item' : (agb.types.BitfieldType('u32', [
                    ('item', 'items', 16),
                    ('flag', None, 8),
                    ('amount', None, 5),
                    ('chunk', None, 3)
                ]), [0, 0, 0, 0]),
                'script' : (backend.OwScriptPointerType(), '0')
            }, self.strucutre_get), None)
        ])
    
    def strucutre_get(self):
        """ Retrives the structure type of the signpost. 
        
        Returns:
        --------
        structure_type : 'item' or 'script' 
            The structure type of the signpost.
        """
        if self.type < 5:
            return 'script'
        else:
            return 'item'