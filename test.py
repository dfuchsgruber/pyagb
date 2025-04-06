from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabBar, QLabel


class TabBarWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        # Create a tab bar
        self.tab_bar = QTabBar()
        self.tab_bar.addTab('Tab 1')
        self.tab_bar.addTab('Tab 2')
        self.tab_bar.addTab('Tab 3')

        # Shared content widget
        self.content = QLabel('Content for Tab 1')

        layout.addWidget(self.tab_bar)
        layout.addWidget(self.content)

        # Connect tab changes
        self.tab_bar.currentChanged.connect(self.update_content)

    def update_content(self, index):
        self.content.setText(f'Content for Tab {index + 1}')


app = QApplication([])
window = TabBarWidget()
window.show()
app.exec()
