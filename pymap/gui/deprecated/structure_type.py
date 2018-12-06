import pyqtgraph.parametertree.parameterTypes as parameterTypes
from agb.types import Structure

import pymap.gui.model.map

""" Model structure types as parameters """

class StructureTypeParameter(parameterTypes.GroupParameter):
    
    def __init__(self, name, project, datatype_name, value):
        """ Initializes the ScalarType Parameter class.
        
        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : dict
            The values of the structure.
        """
        self.datatype = project.model[datatype_name]
        if not isinstance(self.datatype, Structure):
            raise RuntimeError(f'Expected a Structure but got {type(self.datatype)} instead.')
        self.project = project
        super().__init__(name=name)
        # Add all children
        for name, type_name in self.datatype.structure:
            subtype = self.project.model[type_name]
            child = pymap.gui.model.map.type_to_parameter[type(subtype)](name=name, value=value[name])
            self.addChild(child)
