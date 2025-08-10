"""History of opened headers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator, TypedDict

if TYPE_CHECKING:
    from .gui import PymapGui


class OpenHistoryItem(TypedDict):
    """Item for the history of openend headers."""

    project_path: str
    bank: str
    map_idx: str
    label: str


class OpenHistory:
    """History of opened headers."""

    max_size: int = 10

    def __init__(self, main_gui: PymapGui):
        """Initialize the history."""
        self.main_gui = main_gui

    def __iter__(self) -> Generator[OpenHistoryItem]:
        """Iterate over the history."""
        return iter(
            self.main_gui.settings.value('open_history/items', [], list)[  # type: ignore
                : self.max_size
            ]  # type: ignore
        )  # type: ignore

    def __len__(self) -> int:
        """Return the length of the history."""
        return len(self.main_gui.settings.value('open_history/items', [], list))  # type: ignore

    def __getitem__(self, index: int) -> OpenHistoryItem:
        """Get an item from the history."""
        items = self.main_gui.settings.value('open_history/items', [], list)  # type: ignore
        items: list[OpenHistoryItem] = items  # type: ignore
        if index < 0:
            index += len(items)
        if index < 0 or index >= len(items):
            raise IndexError('Index out of range')
        return items[index]  # type: ignore

    def add(self, item: OpenHistoryItem):
        """Add an item to the history.

        Args:
            item: The item to add.
        """
        items = self.main_gui.settings.value('open_history/items', [], list)  # type: ignore
        items: list[OpenHistoryItem] = items  # type: ignore
        items = [item] + items[: self.max_size - 1]  # type: ignore
        # remove duplicates from the list `items` while preserving the order

        seen: set[tuple[tuple[str, object], ...]] = set()
        items = [
            x
            for x in items
            if tuple(sorted(x.items())) not in seen
            and not seen.add(tuple(sorted(x.items())))
        ]

        self.main_gui.settings.setValue('open_history/items', items)  # type: ignore

    def clear(self):
        """Clear the history."""
        self.main_gui.settings.setValue('open_history/items', [])
