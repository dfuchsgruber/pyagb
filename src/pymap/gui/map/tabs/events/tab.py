"""A tab for an event type."""

from __future__ import annotations

from typing import TYPE_CHECKING, SupportsInt

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QMessageBox,
    QPushButton,
    QWidget,
)

from pymap.configuration import PymapEventConfigType
from pymap.gui import properties
from pymap.gui.icon import Icon, icon_paths

from ....history import AppendEvent, RemoveEvent
from .properties import EventProperties

if TYPE_CHECKING:
    from .events_tab import EventsTab


class EventTab(QWidget):
    """Tab for an event type."""

    def __init__(
        self,
        events_tab: EventsTab,
        event_name: str,
        event_type: PymapEventConfigType,
        parent: QWidget | None = None,
    ):
        """Initializes the event tab.

        Args:
            events_tab (EventWidget): The event widget.
            event_name (str): The name of the event.
            event_type (PymapEventConfigType): The event type.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.events_tab = events_tab
        self.event_type = event_type

        layout = QGridLayout()
        self.setLayout(layout)
        self.idx_combobox = QComboBox()
        layout.addWidget(self.idx_combobox, 1, 1)
        self.add_button = QPushButton()

        self.add_button.setIcon(QIcon(icon_paths[Icon.PLUS]))
        self.add_button.clicked.connect(self.append_event)
        layout.addWidget(self.add_button, 1, 2)
        self.remove_button = QPushButton()
        self.remove_button.setIcon(QIcon(icon_paths[Icon.REMOVE]))

        self.remove_button.clicked.connect(self.remove_current_event)

        layout.addWidget(self.remove_button, 1, 3)
        self.event_properties = EventProperties(self, event_name)
        layout.addWidget(self.event_properties, 2, 1, 1, 3)
        if event_type.get('goto_header_button_button_enabled', False):
            self.goto_header_button = QPushButton('Go to target header')
            self.goto_header_button.clicked.connect(self.goto_current_header)
            layout.addWidget(self.goto_header_button, 3, 1, 1, 3)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)
        self.idx_combobox.currentIndexChanged.connect(self.select_event)

    def _sanitize_event_idx(self, event_idx: int) -> int:
        """Sanitizes an event index.

        Args:
            event_idx (int): The event index.

        Returns:
            int: The sanitized event index.
        """
        if (
            self.events_tab.map_widget.main_gui.project is None
            or self.events_tab.map_widget.main_gui.header is None
        ):
            return -1
        number_events = properties.get_member_by_path(
            self.events_tab.map_widget.main_gui.header, self.event_type['size_path']
        )
        assert isinstance(number_events, SupportsInt)
        number_events = int(number_events)

        # If -1 is selcted, select first, but never select a no more present event
        return min(number_events - 1, max(0, event_idx))

    def select_event(self):
        """Selects the event of the current index."""
        self.event_properties.load()
        self.events_tab.map_widget.map_scene.update_selected_event_image(
            self.event_type, self._sanitize_event_idx(self.idx_combobox.currentIndex())
        )

    def load_events(self):
        """Updates the events according to the model."""
        self.event_properties.clear()
        if (
            self.events_tab.map_widget.main_gui.project is None
            or self.events_tab.map_widget.main_gui.header is None
        ):
            self.idx_combobox.blockSignals(True)
            self.idx_combobox.clear()
            self.idx_combobox.blockSignals(False)
        else:
            # Load events
            events = self.events_tab.map_widget.main_gui.get_events(self.event_type)
            assert isinstance(events, list)

            number_events = properties.get_member_by_path(
                self.events_tab.map_widget.main_gui.header, self.event_type['size_path']
            )
            assert isinstance(number_events, SupportsInt)
            number_events = int(number_events)
            current_idx = self._sanitize_event_idx(self.idx_combobox.currentIndex())

            self.idx_combobox.blockSignals(True)
            self.idx_combobox.clear()
            self.idx_combobox.addItems(list(map(str, range(number_events))))
            self.idx_combobox.setCurrentIndex(current_idx)
            # We want select event to be triggered even if the current idx is -1 in
            # order to clear the properties
            self.select_event()
            self.idx_combobox.blockSignals(False)

    def remove_current_event(self):
        """Removes the current event."""
        self.remove_event(self.idx_combobox.currentIndex())

    def remove_event(self, event_idx: int):
        """Removes an event."""
        if (
            self.events_tab.map_widget.main_gui.project is None
            or self.events_tab.map_widget.main_gui.header is None
        ):
            return

        if event_idx < 0:
            return

        self.events_tab.map_widget.undo_stack.push(
            RemoveEvent(self.events_tab, self.event_type, event_idx)
        )

    def append_event(self):
        """Appends a new event."""
        if (
            self.events_tab.map_widget.main_gui.project is None
            or self.events_tab.map_widget.main_gui.header is None
        ):
            return
        self.events_tab.map_widget.undo_stack.push(
            AppendEvent(self.events_tab, self.event_type)
        )

    def goto_current_header(self):
        """Goes to the header associated with the current event."""
        self.goto_header(self.idx_combobox.currentIndex())

    def goto_header(self, event_idx: int):
        """Goes to a new header associated with an event."""
        if (
            self.events_tab.map_widget.main_gui.project is None
            or self.events_tab.map_widget.main_gui.header is None
            or event_idx < 0
        ):
            return

        event = self.events_tab.map_widget.main_gui.get_event(
            self.event_type, event_idx
        )
        assert 'target_bank_path' in self.event_type, (
            'target_bank_path not found in event_type'
        )
        assert 'target_map_idx_path' in self.event_type, (
            'target_map_idx_path not found in event_type'
        )

        target_bank = properties.get_member_by_path(
            event, self.event_type['target_bank_path']
        )
        target_map_idx = properties.get_member_by_path(
            event, self.event_type['target_map_idx_path']
        )

        target_warp_idx = int(
            str(
                properties.get_member_by_path(
                    event, self.event_type['target_warp_idx_path']
                )
            ),
            0,
        )
        try:
            self.events_tab.map_widget.main_gui.open_header(
                str(target_bank), str(target_map_idx)
            )

        except KeyError as e:
            return QMessageBox.critical(
                self,
                'Header can not be opened',
                f'The header {target_bank}.{target_map_idx} could not be opened. '
                f'Key {e.args[0]} was not found.',
            )

        # If possible, we also want to select the appropriate warp.#
        # TODO: this is probably only a semi-good idea, it assumes that again
        # we are directed to the warps tab
        number_events = properties.get_member_by_path(
            self.events_tab.map_widget.main_gui.header, self.event_type['size_path']
        )
        assert isinstance(number_events, SupportsInt)
        number_events = int(number_events)

        current_idx = self._sanitize_event_idx(target_warp_idx)
        self.idx_combobox.blockSignals(True)
        self.idx_combobox.setCurrentIndex(current_idx)
        # We want select event to be triggered even if the current idx is -1 in
        # order to clear the properties
        self.select_event()
        self.idx_combobox.blockSignals(False)
