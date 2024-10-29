Interactive Data Plotting Tool with PyQt5 and Plotly

This project provides a Python-based graphical interface for creating interactive data visualizations using PyQt5 and Plotly. Designed as a flexible tool for exploring datasets, this application allows users to select data columns, apply various plot types, and view the resulting visualizations directly in the GUI. With the ability to load CSV or Excel files and a modular approach to plotting, this tool is suitable for data analysts, students, and researchers who want to analyze trends, distributions, and patterns without coding Plotly visuals from scratch.

Key Features

	•	Interactive GUI: An easy-to-use interface to customize x-axis, y-axis, and plot types.
	•	Dynamic Data Loading: Upload new datasets directly in the GUI, supporting CSV and Excel formats.
	•	Customizable Plot Options: Choose from different plot types provided via the PlotMeta module for more targeted data analysis.
	•	Plotly Integration: Generate interactive plots rendered in a PyQt5 web view, with HTML output for flexibility.

Usage

This tool is ideal for data visualization tasks, such as:

	•	Quick exploratory data analysis (EDA)
	•	Identifying trends and patterns in datasets
	•	Educational purposes, where interactive visuals can enhance understanding
	•	Use cases requiring rapid prototyping of data visuals with minimal setup

Simply run the application, load your dataset, and start exploring your data with interactive, customizable visualizations.

--------------------------------------------
How to launch from Jupyter Notebook with your data:

'''
import subprocess

chartly_path = path_to_app
subprocess.Popen(['python3', f'{chartly_path}app.py', '--file', f'{chartly_path}your_dataset.csv'])
'''

--------------------------------------------

There are some files and functions for my future ideas.