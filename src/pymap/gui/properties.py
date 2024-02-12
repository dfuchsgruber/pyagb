
import pyqtgraph.parametertree.parameterTypes as parameterTypes
from agb.model.type import ModelParents, ModelValue
from agb.types import *
from pyqtgraph.Qt import *


class ConstantComboBox(QtWidgets.QComboBox):
    """Subclass this thing in order to manually filter out undo events."""

    def event(self, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.matches(QtGui.QKeySequence.Undo) or event.matches(QtGui.QKeySequence.Redo):
                return False
        return super().event(event)

# Parameter item for parameters associated with constants
class ConstantParameterItem(parameterTypes.WidgetParameterItem):
    """Inherit from the standard list parameter item class but make the widget editable."""

    def __init__(self, param, depth):
        self.constants = param.constants
        super().__init__(param, depth)

    def makeWidget(self):
        opts = self.param.opts
        t = opts['type']
        w = ConstantComboBox()
        w.setMaximumHeight(20)  ## set to match height of spin box and line edit
        w.sigChanged = w.editTextChanged
        w.setValue = self.setValue
        w.value = w.currentText
        w.setEditable(True)
        w.setContextMenuPolicy(0) # No context menu
        self.widget = w
        w.addItems(self.constants)
        return w

    def setValue(self, val):
        #print(f'Set {val}')
        self.widget.setEditText(str(val))


# Base class for scalar types associated with constants as well as children of bitfields with constants
class ConstantsTypeParameter(parameterTypes.ListParameter):

    itemClass = ConstantParameterItem

    def __init__(self, name, project, constants, **kwargs):
        self.project = project
        self.constants = constants
        super().__init__(name=name, values=constants, **kwargs)


    def model_value(self):
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : str
            The value of the parameter.
        """
        return self.value()

    def update(self, value):
        """Updates this parameter."""
        self.setValue(value)

# Scalar parameter that is associated with constants
class ScalarTypeParameter(ConstantsTypeParameter):

    # Parameter for the tree that builds upon a scalar type
    def __init__(self, name, project, datatype_name, value, context, model_parent, **kwargs):
        """Initializes the ScalarType Parameter class.

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
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        self.datatype = project.model[datatype_name]
        self.project = project
        self.context = context
        self.model_parent = model_parent
        # Make constants appear in the combo box
        if getattr(self.datatype, 'constant', None) is not None:
            constants = [value for value in self.project.constants[self.datatype.constant]]
        else:
            constants = []
        super().__init__(name, self.project, constants, **kwargs)
        self.setValue(value)


""" Model structure types as parameters """

class StructureTypeParameter(parameterTypes.GroupParameter):

    def __init__(self, name, project, datatype_name, value, context, model_parent, typecheck=True, **kwargs):
        """Initializes the Structure Parameter class.

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
        context : list
            The context.
        model_parents : parameterTypes.Parameter
            The parents of the parameter according to the data model.
        """
        self.datatype = project.model[datatype_name]
        self.project = project
        self.context = context
        self.model_parent = model_parent
        super().__init__(name=name, **kwargs)
        # Add all children
        for name, type_name, _ in sorted(self.datatype.structure, key=lambda x: x[2]):
            if name not in self.datatype.hidden_members:
                child = type_to_parameter(project, type_name)(name, project, type_name, value[name], context + [name], self)
                self.addChild(child)

    def model_value(self):
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : dict
            The value of the parameter.
        """
        return { name : self.child(name).model_value() for name, _, _ in self.datatype.structure if name not in self.datatype.hidden_members}

    def update(self, value):
        """Recursively updates the values of all children."""
        for name, type_name, _ in sorted(self.datatype.structure, key=lambda x: x[2]):
            if name not in self.datatype.hidden_members:
                self.child(name).update(value[name])


class BitfieldTypeParameter(parameterTypes.GroupParameter):

    def __init__(self, name, project, datatype_name, value, context, model_parent, **kwargs):
        """Initializes the Bitifield Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : dict
            The values of the bitfield.
        context : list
            The context.
        model_parents : parameterTypes.Parameter
            The parents of the parameter according to the data model.
        """
        self.datatype = project.model[datatype_name]
        self.project = project
        self.context = context
        self.model_parent = model_parent
        super().__init__(name=name, **kwargs)
        # Add all children
        for name, constant, size in self.datatype.structure:
            if constant is not None:
                child = ConstantsTypeParameter(name, self.project, list(self.project.constants[constant]))
                child.setValue(value[name])
            else:
                child = ScalarTypeParameter(name, self.project, datatype_name, value[name], self.context + [name], self)
            if name not in self.datatype.hidden_members: self.addChild(child)

    def model_value(self):
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : dict
            The value of the parameter.
        """
        return { name : self.child(name).model_value() for name, _, _ in self.datatype.structure }

    def update(self, value):
        """Recursively updates the values of all children."""
        for name, constant, size in self.datatype.structure:
            self.child(name).update(value[name])


class ArrayTypeParameter(parameterTypes.GroupParameter):

    def __init__(self, name, project, datatype_name, values, context, model_parent, typecheck=True, **kwargs):
        """Initializes the Structure Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : list
            The values of the array.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        self.datatype = project.model[datatype_name]
        self.project = project
        self.context = context
        self.model_parent = model_parent
        super().__init__(name=name, **kwargs)
        # Add the list that enbales navigating through the array
        size = self.size_get()
        self.addChild(parameterTypes.ListParameter(name='idx', title='Index', values=list(range(size)), value=None, default=None))
        # Create one parameter for each value
        self.values = []
        for idx, value in enumerate(values):
            self._insert(idx, value=value)
        self.update_value()

    def _insert(self, idx, value=None):
        """Inserts a new element into the array.

        Parameters:
        -----------
        idx : int or str or None
            The index to insert the element at.
        value : object or None
            The value to insert. If None a new value will be created.
        """
        if idx is not None and idx != '' and int(idx) in range(self.size_get() + 1):
            value_datatype_name = self.datatype.datatype
            value_parameter_class = type_to_parameter(self.project, value_datatype_name)
            if value is None:
                value = self.project.model[value_datatype_name](self.project, self.context + [int(idx)], model_parents(self) + [self.model_value()])
            self.values.insert(idx, value_parameter_class(str(idx), self.project, value_datatype_name, value, self.context + [idx], self, title=f'<{value_datatype_name}>'))

    def treeStateChanged(self, param, changes):
        super().treeStateChanged(param, changes)
        if param is self.child('idx'):
            # Load the element at this index
            self.update_value()

    def update_value(self):
        # Remove all subtree children
        for subtree in self.values:
            if subtree.parent() is not None:
                assert(subtree.parent() is self)
                subtree.remove()
        # Only show the child that matches the current index
        idx = self.child('idx').value()
        if idx is not None and idx != '' and int(idx) in range(self.size_get()):
            self.addChild(self.values[int(idx)])

    def model_value(self):
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : list
            The value of the parameter.
        """
        return list(map(lambda child: child.model_value(), self.values))

    def update(self, values):
        """Updates the values in the array."""
        for child, value in zip(self.values, value):
            child.update(value)

class FixedSizeArrayTypeParameter(ArrayTypeParameter):

    def __init__(self, name, project, datatype_name, value, context, model_parent, **kwargs):
        """Initializes the Array Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : list
            The values of the array.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        super().__init__(name, project, datatype_name, value, context, model_parent, **kwargs)

    def size_get(self):
        return self.datatype.size_get(self.project, self.context, None)


class VariableSizeArrayTypeParameter(ArrayTypeParameter):

    def __init__(self, name, project, datatype_name, value, context, model_parent, **kwargs):
        """Initializes the Array Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : list
            The values of the array.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        super().__init__(name, project, datatype_name, value, context, model_parent, **kwargs)
        if not isinstance(self.datatype, VariableSizeArrayType) and not isinstance(self.datatype, UnboundedArrayType):
            raise RuntimeError(f'Expected a VariableSizeArrayType or UnboundedArrayType but got {type(self.datatype)} instead.')
        # Add widgets to add and remove elements
        self.addChild(parameterTypes.ActionParameter(name='Remove'))
        self.child('Remove').sigActivated.connect(lambda: self._remove(idx=self.child('idx').value()))
        self.addChild(parameterTypes.ActionParameter(name='Append'))
        self.child('Append').sigActivated.connect(self._append)
        # Make length in parent read-only
        if self._size_location() is not None:
            self._size_location().setReadonly(True)
        self.update_value()

    def _adaptLimits(self):
        # Assert size still matches
        if self.size_get() != len(self.values):
            raise RuntimeError(f'Size mismatch. Parent uses {self.size_get()} but array only holds {len(self.values)}')
        self.child('idx').setLimits(list(range(len(self.values))))

    def _append(self):
        """Appends a default element to the array."""
        size = self.size_get()
        self._insert(size)
        # Increment size
        self.size_set(size + 1)
        # Change limits
        self._adaptLimits()

    def _remove(self, idx):
        """Removes an element from the array.

        Parameters:
        -----------
        idx : int or str or None
            The index to remove.
        """
        if idx is not None and idx != '' and int(idx) in range(self.size_get()):
            idx = int(idx)
            self.values[idx].remove()
            del self.values[idx]
            self.size_set(len(self.values))
            self._adaptLimits()
            self.child('idx').setValue(len(self.values) - 1)
            #self.update_value()


    def _size_location(self):
        # Climb up
        n_parents, location = self.datatype.size_path
        if n_parents <= 0:
            raise RuntimeError(f'Upwards parent traversals must be positive, not {n_parents}')
        root = self
        for _ in range(n_parents):
            root = root.model_parent
        for member in location:
            root = root.child(member)
        return root

    def size_get(self):
        return self.datatype.size_cast(self._size_location().value(), self.project)

    def size_set(self, size):
        self._size_location().setValue(size)

class UnboundedArrayTypeParameter(VariableSizeArrayTypeParameter):

    def __init__(self, name, project, datatype_name, value, context, model_parent, **kwargs):
        """Initializes the Array Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : list
            The values of the array.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        self.values = [None] * len(value) # Such that the size becomes available
        super().__init__(name, project, datatype_name, value, context, model_parent, **kwargs)

    def size_get(self):
        return len(self.values)

    def size_set(self, size):
        pass

    def _size_location(self):
        pass


class PointerParameter(parameterTypes.GroupParameter):

    add_reference = 'Add reference' # Name and text of the 'Add reference' button
    remove_reference = 'Remove reference' # Name and text of the 'Remove reference' button
    referred = 'referred' # Name of the subtree that holds the referred values

    def __init__(self, name, project, datatype_name, value, context, model_parent, **kwargs):
        """Initializes the Pointer Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        value : object or None
            The value of the refered data or None if the pointer points to null.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        self.datatype = project.model[datatype_name]
        self.project = project
        self.context = context
        self.model_parent = model_parent
        super().__init__(name=name, **kwargs)
        if value is not None:
            self.add_new(referred=value)
        else:
            self._add_add()

    def add_new(self, referred=None):
        """Adds a new instance of a refered value if currently null is referred, i.e. no child is held by this group.

        Parameters:
        -----------
        value : object or None
            The value to add. If None the datatype is default initialized.
        """
        referred_datatype_name = self.datatype.datatype
        referred_parameter_class = type_to_parameter(self.project, referred_datatype_name)
        if referred is None:
            referred = self.project.model[referred_datatype_name](self.project, self.context, model_parents(self))
        self.addChild(referred_parameter_class(PointerParameter.referred, self.project, referred_datatype_name, referred, self.context, self.model_parent, removable=True, title=f'Reference to <{referred_datatype_name}>'))
        # Remove the add button if present
        try:
            self.child(PointerParameter.add_reference).remove()
        except Exception:
            pass
        # child_remove = parameterTypes.ActionParameter(name=PointerParameter.remove_reference)
        # child_remove.sigActivated.connect(self.remove)
        # self.addChild(child_remove)

    def _add_add(self):
        child_add = parameterTypes.ActionParameter(name=PointerParameter.add_reference)
        child_add.sigActivated.connect(lambda: self.add_new(referred=None))
        self.addChild(child_add)

    def treeStateChanged(self, param, changes):
        super().treeStateChanged(param, changes)
        if not self.hasChild():
            # We do not have a child anymore, add the add-button
            self._add_add()

    def remove(self):
        """Removes the instance of the referred value."""
        if self.hasChild():
            self.child(PointerParameter.referred).remove()
        # Remove the delete button if present
        try:
            self.child(PointerParameter.remove_reference).remove()
        except Exception:
            pass

    def hasChild(self):
        """Checks if this parameter currently refers to a child."""
        try:
            self.child(PointerParameter.referred)
            return True
        except:
            return False

    def update(self, value):
        """Updates the pointer reference."""
        if value is None:
            self.remove()
        else:
            if self.hasChild():
                self.child(PointerParameter.referred).update(value)
            else:
                self.add_new(referred=value)


    def model_value(self):
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : dict
            The value of the parameter.
        """
        try:
            return self.child(PointerParameter.referred).model_value()
        except:
            return None

class UnionTypeParameter(parameterTypes.GroupParameter):

    def __init__(self, name, project, datatype_name, values, context, model_parent, **kwargs):
        """Initializes the Union Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : dict
            The values of all union subtypes.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        self.datatype = project.model[datatype_name]
        self.project = project
        self.context = context
        self.model_parent = model_parent
        super().__init__(name=name, **kwargs)
        # Create children for all possible subtypes
        self.values = {}
        for name in self.datatype.subtypes:
            subtype = self.datatype.subtypes[name]
            self.values[name] = type_to_parameter(self.project, subtype)(f'{subtype}', self.project, subtype, values[name], context + [name], self, title=f'View as <{subtype}>')
            self.addChild(self.values[name])
        #self.update_value()

    def update_value(self):
        """Displays the correct union subtype."""
        # Get the active name
        active_name = self.datatype.name_get(self.project, self.context, model_parents(self))
        print(f'Parent changed to {active_name}')
        for name in self.values:
            child = self.values[name]
            if name == active_name and child.parent() is not self:
                self.addChild(child)
            elif name != active_name and child.parent() is self:
                child.remove()


    def model_value(self):
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : dict
            The values of the parameter.
        """
        return {name : self.values[name].model_value() for name in self.values}

    def update(self, value):
        """Updates all children of this parameter."""
        for name in self.values:
            self.child(self.datatype.subtypes[name]).update(value[name])


def model_parents(root):
    """Returns the parent values of a parameter in the serializable model format instead of a tree.

    Parameters:
    -----------
    root : Parameter
        The paremeter of which the parents will be retrieved.

    Returns:
    --------
    parents : list
        The parents of this value, with the last being the direct ancestor.
    """
    parents = []
    while root.model_parent is not None:
        root = root.model_parent
        parents = [root.model_value()] + parents
    return parents

def type_to_parameter(project, datatype_name):
    """Translates a datatype into a parameter class.

    Parameters:
    -----------
    project : pymap.project.Project
        The pymap project.
    datatype_name : str
        The name of the datatype.

    Returns:
    --------
    parameter_class : parameterTypes.Parameter
        The corresponding parameter class.
    """
    datatype = project.model[datatype_name]
    if isinstance(datatype, DynamicLabelPointer):
        raise NotImplementedError('Dynamic label pointers not yet supported.')
    elif isinstance(datatype, PointerType):
        return PointerParameter
    elif isinstance(datatype, BitfieldType):
        return BitfieldTypeParameter
    elif isinstance(datatype, ScalarType):
        return ScalarTypeParameter
    elif isinstance(datatype, Structure):
        return StructureTypeParameter
    elif isinstance(datatype, FixedSizeArrayType):
        return FixedSizeArrayTypeParameter
    elif isinstance(datatype, VariableSizeArrayType):
        return VariableSizeArrayTypeParameter
    elif isinstance(datatype, UnboundedArrayType):
        return UnboundedArrayTypeParameter
    elif isinstance(datatype, UnionType):
        return UnionTypeParameter
    else:
        raise RuntimeError(f'Unsupported datatype class {type(datatype)} of {datatype}')

def get_member_by_path(value: ModelValue, path: list[str | int]) -> ModelValue:
    """Returns an attribute of a structure by its path."""
    for edge in path:
        value = value[edge]
    return value

def set_member_by_path(target: ModelValue, value: ModelValue,
                       path: list[str | int]):
    """Sets the value of a structure by its path.

    Parameters:
    -----------
    target : dict
        The structure that holds the requested value
    value : str
        The value to apply
    path : list
        A path to access the attribute
    """
    for edge in path[:-1]:
        match target:
            case list() if isinstance(edge, int):
                target: ModelValue = target[edge] # type: ignore
            case dict() if isinstance(edge, (str)):
                target: ModelValue = target[edge] # type: ignore
            case _: # type: ignore
                raise RuntimeError(f'Unsupported edge type {type(edge)}')
    assert isinstance(target, (dict, list))
    target[path[-1]] = value # type: ignore

def get_parents_by_path(value: ModelValue, path: list[str | int]) -> ModelParents:
    """Builds the parents of an instance based on its path.

    Note that the requested data instance is not needed to be present
    for this method to work. Just all its parent have to be.

    Parameters:
    -----------
    value : dict
        The origin structure that contains a data instance.
    path : list
        A path to access the data instance.

    Returns:
    --------
    parents : list
        The model parents of this data instance.
    """
    parents = [value]
    for member in path[:-1]:
        value = value[member]
        parents.append(value)
    return parents
