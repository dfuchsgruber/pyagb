"""Abstract Type class."""

from __future__ import annotations

import ast
import operator
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypeAlias
from warnings import warn

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pymap.project import Project

ScalarModelValue: TypeAlias = int | str | bool | None
IntArray: TypeAlias = npt.NDArray[np.int_]
ModelValue: TypeAlias = (
    ScalarModelValue
    | Sequence['ModelValue']
    | dict[str, 'ModelValue']
    | tuple[str, 'ModelValue']
    | IntArray
    | npt.NDArray[np.uint8]
)
ModelContextItem: TypeAlias = int | str | bool
ModelContext: TypeAlias = Sequence[ModelContextItem]
ModelParents: TypeAlias = Sequence[ModelValue]

# Each project can define models which are mappings from
# type names to the respective type instances.
Model: TypeAlias = Mapping[str, 'Type']


class Type(ABC):
    """Base type class."""

    @abstractmethod
    def from_data(
        self,
        rom: bytearray,
        offset: int,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> ModelValue:
        """Retrieves the type value from a rom.

        Parameters:
        -----------
        rom : bytearray
            The rom to retrieve the data from.
        offset : int
            The offset to retrieve the data from.
        proj : pymap.project.Project
            The pymap project to access e.g. constants.
        context : list of str
            The context in which the data got initialized.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.

        Returns:
        --------
        value : object
        """
        raise NotImplementedError

    @abstractmethod
    def to_assembly(
        self,
        value: ModelValue,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
        label: str | None = None,
        alignment: int | None = None,
        global_label: bool = False,
    ) -> tuple[str, list[str]]:
        """Creates an assembly representation of the type.

        Parameters:
        -----------
        value : object
            The object to assemble.
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list
            Context from parent elements.
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
            The assembly representation of the object.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        raise NotImplementedError

    def __call__(
        self, project: Project, context: ModelContext, parents: ModelParents
    ) -> ModelValue:
        """Initializes a new object.

        Parameters:
        -----------
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list
            Context from parent elements.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are generated depth-first.

        Returns:
        --------
        string : str
            Empty string.
        """
        raise NotImplementedError

    def size(
        self,
        value: ModelValue,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> int:
        """Returns the size of the object.

        Parameters:
        -----------
        object : object
            The object
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list
            Context from parent elements.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are generated depth-first.

        Returns:
        --------
        size : int
            The size of this type in bytes.
        """
        raise NotImplementedError

    def get_constants(
        self,
        value: ModelValue,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> set[str]:
        """The set of all constants that are used by this type and potential subtypes.

        Parameters:
        -----------
        object : object
            The object.
        project : pymap.project.Project
            The project to e.g. fetch constants from
        context : list
            Context from parent elements.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are generated depth-first.

        Returns:
        --------
        constants : set of str
            A set of all required constants.
        """
        raise NotImplementedError


def associate_with_constant(
    value: ScalarModelValue, proj: Project, constant: str | None
) -> ScalarModelValue | str:
    """Tries to associate a value form a constant table of the pymap project.

    Parameters:
    -----------
    value : int
        The value to associate with a constant
    proj : pymap.project.Project
        The pymap project to contain the constant tables.
    constant : str or None
            The constant table this type will be associated with.

    Returns:
    --------
    associated : int or str
        The association of the value or the original value if no association
        was possible.
    """
    if constant is not None:
        if constant in proj.constants:
            constants = proj.constants[constant]
            for key in constants:
                if constants[key] == value:
                    return key
            warn(f'No match for value {value} found in constant table {constant}')
        else:
            warn(f'Constant table {constant} not found in project.')
    assert isinstance(value, (ScalarModelValue, str))
    return value


def label_and_align(
    assembly: str, label: str | None, alignment: int | None, global_label: bool
) -> str:
    """Adds label and alignment to an assembly representation of a type.

    Parameters:
    -----------
    assembly : str
        The assembly representation of the type
    label : str or None
        The label of the type if requested
    alignment : int or None
        The alignment of the type if requested
    global_label : bool
        If the label generated will be exported globally.
        Only relevant if label is not None.

    Returns:
    --------
    assembly : str
        The aligned and labeled assembly representation of the type
    """
    blocks: list[str] = []
    if alignment is not None:
        blocks.append(f'.align {alignment}')
    if label is not None:
        if global_label:
            blocks.append(f'.global {label}')
        blocks.append(f'{label}:')
    blocks.append(assembly)
    return '\n'.join(blocks)


def evaluate_expression_with_constants(  # noqa: C901
    expression: str, proj: Project, *constant_tables: str
) -> int | float | str:
    """Safely evaluates a Python-like expression with constants from project.

    Parameters:
    -----------
    expression : str
        The expression to evaluate (e.g., "3 + FOO", "BAR * 2")
    proj : pymap.project.Project
        The pymap project containing constant tables
    *constant_tables : str
        Variable number of constant table names to load

    Returns:
    --------
    result : int | float | str
        The evaluated result of the expression
    """
    # Collect all constants from specified tables
    constants: dict[str, int] = {}
    for table_name in constant_tables:
        if table_name in proj.constants:
            constants.update(proj.constants[table_name])
        else:
            warn(f'Constant table {table_name} not found in project.')

    # Safe operators and functions we allow
    allowed_operators: dict[Any, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Invert: operator.invert,
    }

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):  # Numbers, strings
            return node.value
        elif isinstance(node, ast.Name):  # Variable names (constants)
            if node.id in constants:
                return constants[node.id]
            else:
                raise NameError(
                    f"Constant '{node.id}' not found in any of the specified tables: "
                    f'{constant_tables}'
                )
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_func = allowed_operators.get(type(node.op))
            if op_func is None:
                raise ValueError(
                    f'Unsupported binary operator: {type(node.op).__name__}'
                )
            return op_func(left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = _eval_node(node.operand)
            op_func = allowed_operators.get(type(node.op))
            if op_func is None:
                raise ValueError(
                    f'Unsupported unary operator: {type(node.op).__name__}'
                )
            return op_func(operand)
        elif isinstance(node, ast.Expression):
            return _eval_node(node.body)
        else:
            raise ValueError(f'Unsupported expression type: {type(node).__name__}')

    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        # Evaluate the AST safely
        return _eval_node(tree)
    except SyntaxError as e:
        raise ValueError(f'Invalid expression syntax: {expression}') from e
