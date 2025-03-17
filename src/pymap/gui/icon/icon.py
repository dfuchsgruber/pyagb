"""Icon resources for the resource tree."""

import importlib.resources as resources
from enum import StrEnum, unique


@unique
class Icon(StrEnum):
    """Enum for the icons."""

    HEADER = 'header'
    FOLDER = 'folder'
    TREE = 'tree'
    FOOTER = 'footer'
    TILESET = 'tileset'
    GFX = 'gfx'
    PLUS = 'plus'
    REMOVE = 'remove'
    IMPORT = 'import'
    RENAME = 'rename'
    TAG = 'tag'
    RELOAD = 'reload'
    EDIT = 'edit'
    CLEAR = 'clear'


# Icon paths relative to the resources 'icon'
icon_paths: dict[Icon, str] = {
    Icon.HEADER: str(
        resources.files('pymap.gui.icon').joinpath('project_tree_header.png')
    ),
    Icon.FOLDER: str(
        resources.files('pymap.gui.icon').joinpath('project_tree_folder.png')
    ),
    Icon.TREE: str(resources.files('pymap.gui.icon').joinpath('project_tree_tree.png')),
    Icon.FOOTER: str(
        resources.files('pymap.gui.icon').joinpath('project_tree_footer.png')
    ),
    Icon.TILESET: str(
        resources.files('pymap.gui.icon').joinpath('project_tree_tileset.png')
    ),
    Icon.GFX: str(resources.files('pymap.gui.icon').joinpath('project_tree_gfx.png')),
    Icon.PLUS: str(resources.files('pymap.gui.icon').joinpath('plus.png')),
    Icon.REMOVE: str(resources.files('pymap.gui.icon').joinpath('remove.png')),
    Icon.IMPORT: str(resources.files('pymap.gui.icon').joinpath('import.png')),
    Icon.RENAME: str(resources.files('pymap.gui.icon').joinpath('rename.png')),
    Icon.TAG: str(resources.files('pymap.gui.icon').joinpath('tag.png')),
    Icon.RELOAD: str(resources.files('pymap.gui.icon').joinpath('reload.png')),
    Icon.EDIT: str(resources.files('pymap.gui.icon').joinpath('edit.png')),
    Icon.CLEAR: str(resources.files('pymap.gui.icon').joinpath('clear.png')),
}
