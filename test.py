from PySide6.QtWidgets import QApplication, QComboBox, QMainWindow, QWidget, QLabel
import sys
from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtGui import QKeyEvent, QAction, QKeySequence, QUndoStack, QUndoCommand


class QComboBoxWithoutUndo(QComboBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride:
            if event.matches(QKeySequence.StandardKey.Undo):
                return True
        return super().eventFilter(watched, event)
        # print(super().eventFilter(watched, event), event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Main Window with Undo')
        self.setGeometry(100, 100, 400, 300)

        self.undo_stack = QUndoStack(self)

        self.init_ui()

    def init_ui(self):
        self.comboBox = QComboBoxWithoutUndo(self)
        self.comboBox.addItem('Option 1')
        self.comboBox.addItem('Option 2')
        self.comboBox.addItem('Option 3')
        self.setCentralWidget(self.comboBox)

        # Create an action for undo
        undo_action = QAction('Undo', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.undo_stack.undo)
        self.addAction(undo_action)

        # Connect the currentIndexChanged signal to a slot in the main window

        # Connect the currentIndexChanged signal to a slot in the main window
        self.comboBox.editTextChanged.connect(self.combo_text_changed)
        self.installEventFilter(self)

    def combo_text_changed(self, text):
        # Add the current text to the undo stack
        self.undo_stack.push(ComboTextChangedCommand(self.comboBox, text))
        print(self.undo_stack.index())

    def eventFilter(self, obj, event):
        # if event.type() == QEvent.Type.KeyPress:
        #     if (
        #         event.key() == Qt.Key.Key_Z
        #         and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        #     ):
        #         self.undo_stack.undo()
        #         return True
        # if event.type() == QEvent.Type.ShortcutOverride:
        #     print('shortcut', event)
        # print('Event:', event)
        return super().eventFilter(obj, event)


class ComboTextChangedCommand(QUndoCommand):
    def __init__(self, combo_box, new_text):
        super().__init__()
        self.combo_box = combo_box
        self.old_text = combo_box.currentText()
        self.new_text = new_text

    def redo(self):
        print('Redo')
        self.combo_box.setEditText(self.new_text)

    def undo(self):
        print('Undo')
        self.combo_box.setEditText(self.old_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
