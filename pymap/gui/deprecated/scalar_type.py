import pyqtgraph.parametertree.parameterTypes as parameterTypes
from agb.types import ScalarType


""" Model scalar types as parameters """

class ScalarTypeParameterItem(parameterTypes.ListParameterItem):
    """ Inherit from the standard list parameter item class but make the widget editable. """

    def makeWidget(self):
        w = super().makeWidget()
        w.setEditable(True)
        return w

class ScalarTypeParameter(parameterTypes.ListParameter):
    
    itemClass = ScalarTypeParameterItem

    # Parameter for the tree that builds upon a scalar type
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
        values : int or str
            The value of the scalar type.
        """
        self.datatype = project.model[datatype_name]
        if not isinstance(self.datatype, ScalarType):
            raise RuntimeError(f'Expected a ScalarTypeParameter but got {type(self.datatype)} instead.')
        self.project = project
        if self.datatype.constant is not None:
            # Make constants appear in the combo box
            values = [value for value in self.project.constants[self.datatype.constant]]
        else:
            values = []
        super().__init__(name=name, value=value, values=values)
