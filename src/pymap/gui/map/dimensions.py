"""Line edit for dimensions."""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence, QRegularExpressionValidator
from PySide6.QtWidgets import QLineEdit, QWidget


class DimensionLineEdit(QLineEdit):
    """Line edit for dimensions in the format "width, height".

    It only accepts integers and hexadecimal numbers, disables the context menu,
    and filters out undo and redo events.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initializes the dimension line edit.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent=parent)
        self.setValidator(
            QRegularExpressionValidator(
                r'^\s*(0[xX][0-9A-Fa-f]+|\d+)(?:\s*,\s*(0[xX][0-9A-Fa-f]+|\d+))?\s*$'
            )
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setToolTip('Width, Height')
        # Filter out undo and redo events
        self.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Filter out undo and redo events."""
        if event.type() == QEvent.Type.ShortcutOverride:
            assert isinstance(event, QKeyEvent)
            if event.matches(QKeySequence.StandardKey.Undo) or event.matches(
                QKeySequence.StandardKey.Redo
            ):
                return True

        return super().eventFilter(watched, event)
