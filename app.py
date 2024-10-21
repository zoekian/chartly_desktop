# /app.py
import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication
from gui.main_window import PlotWindow

def main():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'age': [23, 25, 22, 21, 27, 30, 35, 40, 38, 29],
        'height': [170, 165, 180, 175, 172, 168, 178, 185, 182, 176],
        'weight': [70, 65, 80, 75, 72, 68, 78, 85, 82, 76]
    })

    # Start the PyQt application
    app = QApplication(sys.argv)
    window = PlotWindow(df)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()