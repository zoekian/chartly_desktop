# /gui/main_window.py
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QComboBox, QPushButton, QLabel, QHBoxLayout, QFileDialog
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from local_code.plot_functions import PlotMeta

class PlotWindow(QMainWindow):
    def __init__(self, df):
        super().__init__()

        self.df = df
        self.plot_funcs = {p.name: p.plot for p in PlotMeta.get_subclasses()}
        # Set up the main window
        self.setWindowTitle('Interactive Plot with Plotly')
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control layout
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        # X axis dropdown
        self.x_label = QLabel("X:")
        self.x_combo = QComboBox()
        self.x_combo.addItems(self.df.columns)
        controls_layout.addWidget(self.x_label)
        controls_layout.addWidget(self.x_combo)

        # Y axis dropdown
        self.y_label = QLabel("Y:")
        self.y_combo = QComboBox()
        self.y_combo.addItems(self.df.columns)
        controls_layout.addWidget(self.y_label)
        controls_layout.addWidget(self.y_combo)

        # Plot function dropdown
        self.plot_label = QLabel("Plot function:")
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(self.plot_funcs.keys())
        controls_layout.addWidget(self.plot_label)
        controls_layout.addWidget(self.plot_combo)

        
        # Button to update the plot
        self.update_button = QPushButton("Update Plot")
        self.update_button.clicked.connect(self.plot)
        controls_layout.addWidget(self.update_button)

        # Button to load a new DataFrame (file upload)
        self.load_button = QPushButton("Add New Data")
        self.load_button.clicked.connect(self.load_dataframe)
        controls_layout.addWidget(self.load_button)

        # Web view for the Plotly plot
        self.web_view = QWebEngineView()
        main_layout.addWidget(self.web_view)


    def plot(self):
        # Get the selected columns
        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        func = self.plot_funcs[self.plot_combo.currentText()]

        # Save the plot as an HTML file
        plot_path = os.path.join(os.getcwd(), "plot.html")
        with open(plot_path, 'w') as f:
            f.write(func(self.df, x_col, y_col))
        
        # Load the HTML file in the QWebEngineView
        self.web_view.setUrl(QUrl.fromLocalFile(plot_path))

    def update_dropdowns(self):
        """Update the X and Y dropdowns based on the current DataFrame."""
        if self.df is not None:
            columns = self.df.columns.tolist()
            self.x_combo.clear()
            self.x_combo.addItems(columns)
            self.y_combo.clear()
            self.y_combo.addItems(columns)

    def load_dataframe(self):
        """Open a file dialog to upload a DataFrame and update the plot."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)", options=options)
        
        if file_name:
            try:
                # Load the file into a pandas DataFrame based on its extension
                if file_name.endswith('.csv'):
                    self.df = pd.read_csv(file_name)
                elif file_name.endswith('.xlsx'):
                    self.df = pd.read_excel(file_name)
                else:
                    raise ValueError("Unsupported file format")

                # Update the dropdowns and plot
                self.update_dropdowns()

            except Exception as e:
                print(f"Error loading file: {e}")
