# it's a template

import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QFileDialog, QPushButton, QVBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Viewer")

        # Create button to load file
        self.button = QPushButton("Load CSV File")
        self.button.clicked.connect(self.load_csv)

        # Create table widget to display data
        self.table = QTableWidget()

        # Arrange widgets
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # Method to load CSV and populate table
    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if file_path:
            df = pd.read_csv(file_path)
            self.populate_table(df)

    # Method to populate table with DataFrame data
    def populate_table(self, df):
        self.table.setRowCount(df.shape[0])
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())