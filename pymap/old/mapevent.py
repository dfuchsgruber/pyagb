MAPEVENT_PERSON = 0
MAPEVENT_WARP = 1
MAPEVENT_TRIGGER = 2
MAPEVENT_SIGN = 3

SIGN_STRUCTURE_SCRIPT = 0
SIGN_STRUCTURE_ITEM = 1

class Map_event:
    """ Superclass to mapevent """
    def __init__(self, type): self.type = type
    def to_dict(self): return self.__dict__

class Map_event_person(Map_event):
    """ Class to modell an map event person"""
    def __init__(self):
        self.target_index = 0;
        self.picture = 0;
        self.field_2, self.field_3 = 0, 0
        self.x, self.y = 0, 0
        self.level = 0
        self.behaviour = 0
        self.behaviour_range = 0
        self.field_b = 0
        self.is_trainer, self.is_trainer_padding = False, 0
        self.field_d = 0
        self.alert_radius = 0
        self.script = "0"
        self.flag = 0
        self.field_16, self.field_17 = 0, 0
        super().__init__(MAPEVENT_PERSON)

class Map_event_sign(Map_event):
    """ Class to modell a map event sign"""
    def __init__(self):
        self.x, self.y = 0, 0
        self.level = 0
        self.sign_type = 0
        self.field_6 = 0
        self.field_7 = 0
        #Each item can virtually store also information that is not meant for its state and remember those even after transformation
        self.item_id = "0"
        self.hidden = "0"
        self.count = "0"
        self.script = "0"
        self._struct_setup()
        super().__init__(MAPEVENT_SIGN)
    
    def _struct_setup(self):
        """ Initializes structure corresponding to sign_type"""
        if self.sign_type in range(0, 5):
            self.structure = SIGN_STRUCTURE_SCRIPT
        else:
            self.structure = SIGN_STRUCTURE_ITEM


        
class Map_event_warp(Map_event):
    """ Class to model a map event warp"""
    def __init__(self):
        self.x, self.y = 0, 0
        self.level = 0
        self.target_warp = 0
        self.target_bank = 0
        self.target_map = 0
        super().__init__(MAPEVENT_WARP)

class Map_event_trigger(Map_event):
    """ Class to model a map event trigger"""
    def __init__(self):
        self.x, self.y = 0, 0
        self.level = 0
        self.field_5 = 0
        self.variable = "0"
        self.value = 0
        self.field_a = 0
        self.field_b = 0
        self.script = "0"
        super().__init__(MAPEVENT_TRIGGER)


def from_dict(d):
    """ Retrieves a map event form a dictrionary """
    type = d["type"]
    if type == MAPEVENT_PERSON: e = Map_event_person()
    elif type == MAPEVENT_SIGN: e = Map_event_sign()
    elif type == MAPEVENT_TRIGGER: e = Map_event_trigger()
    elif type == MAPEVENT_WARP: e = Map_event_warp()
    else: raise Exception("Unkown event type " + str(type))
    for k, v in d.items(): e.__setattr__(k, v)
    return e
