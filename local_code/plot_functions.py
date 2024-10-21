import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

def get_error_html(message):
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h3>{message}</h3>
</body>
</html>
"""

def pearsons_correlation(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculate Pearson's correlation coefficient between two columns of a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    col1 (str): The name of the first column.
    col2 (str): The name of the second column.

    Returns:
    float: The Pearson correlation coefficient between the two columns.
    
    Raises:
    ValueError: If either column is not found in the DataFrame.
    """
    
    # Check if both columns exist in the DataFrame
    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' not found in the DataFrame.")
    if col2 not in df.columns:
        raise ValueError(f"Column '{col2}' not found in the DataFrame.")
    
    # Calculate and return Pearson's correlation coefficient
    correlation = round(df[col1].corr(df[col2]), 4)
    print(f'Correlation between {col1} and {col2} is {correlation}')
    return correlation

def group_by_count(df: pd.DataFrame, x_col: str, y_cols: list, to_print: bool = False) -> dict:
    """
    Do not use. Group by the x and y columns and count occurrences of each unique (x, y) point.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    x_col (str): The name of the x column.
    y_cols (list): List of y column names.
    to_print (bool): Whether to print functions

    Returns:
    dict: A dictionary where keys are y column names and values are grouped DataFrames.
    """
    grouped_data = {}
    
    for y_col in y_cols:
        # Group by x and y, then count occurrences
        group = df.groupby([x_col, y_col]).size().reset_index(name='count')
        grouped_data[y_col] = group
    if to_print:
        display(grouped_data)
    return grouped_data



class PlotMeta(type):
    # A list to store references to all subclasses
    _subclasses = []
    
    # A dictionary to store single instances of each subclass (for singleton behavior)
    _instances = {}

    def __new__(cls, name, bases, attrs):
        # Ensure that every subclass has a 'name' attribute
        if 'name' not in attrs:
            raise TypeError(f"Class '{name}' must have a 'name' attribute.")
        
        # Ensure that every subclass has a 'plot' method
        if 'plot' not in attrs or not callable(attrs['plot']):
            raise TypeError(f"Class '{name}' must have a 'plot(x, y)' method.")
        
        # Create the new class using the parent's __new__ method
        new_class = super().__new__(cls, name, bases, attrs)
        
        # Add the class to the list of subclasses
        cls._subclasses.append(new_class)

        return new_class

    def __call__(cls, *args, **kwargs):
        # If the class already has an instance, return it (singleton behavior)
        if cls not in cls._instances:
            # Create a new instance and store it in the dictionary
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def get_subclasses(cls):
        return cls._subclasses


class LineAndHist(metaclass=PlotMeta):
    name = 'mixed: line and histogram'
    @staticmethod
    def plot(df: pd.DataFrame, x_column: str, y_column: str) -> None:
        '''
        Visualizes the relationship between a feature (x_column) in a DataFrame and a target variable (y_column).
        The function generates a dual-axis plot showing the distribution of the feature values and the mean of the target values for those feature groups.
    
        Parameters:
        df (pd.DataFrame): The pandas DataFrame containing the data to be analyzed.
        x_column (str): The name of the feature (independent variable) column in the DataFrame to plot.
        y_column (str, optional): The name of the target (dependent variable) column in the DataFrame to analyze in relation to the feature. 
            Defaults to 'target'.
        
        Returns: None
        '''
        if (x_column == y_column):
            return get_error_html("X shouldn't be the same as Y")
        
        # Check if the feature column is numeric and compute correlation if applicable
        x_name = col_name_to_title(x_column)
        y_name = col_name_to_title(y_column)
        
        if df[x_column].dtype != object:
            correlation = df[x_column].corr(df[y_column])
            cor_text  = f"Correlation between {x_name} and {y_name}: {correlation:.4f}"
        else:
            cor_text = ""
        
        # Handle high cardinality features (more than 100 unique values)
        if df[x_column].nunique() > 100:
            return get_error_html(f"{x_column} has too many unique values: {df[x_column].nunique()}")
        
        # Group data by the feature to calculate mean target values and counts
        dfgr = df.groupby([x_column])
        df_sorted = dfgr[[y_column]].mean()
        df_sorted['count'] = dfgr[y_column].count()
        df_sorted.reset_index(inplace=True)
        
        # Set up the subplots with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    
        # Bar chart for feature distribution (counts)
        fig.add_trace(
            go.Bar(x=df_sorted[x_column], y=df_sorted['count'], name=f'{x_name} Distribution',
                   marker_color='skyblue', opacity=0.9),
            secondary_y=False
        )
    
        # Scatter plot for target variable mean values
        fig.add_trace(
            go.Scatter(x=df_sorted[x_column], y=df_sorted[y_column], mode='markers+lines',
                       name=f'{y_name} (Mean)', marker=dict(color='orange', size=8)),
            secondary_y=True
        )
    
        # Add titles and axis labels
        fig.update_layout(
            title=f'Distribution of {x_name} and {y_name} over {x_name}<br><sub>{cor_text}</sub>',
            xaxis_title=f'{x_name}',
            yaxis_title=f'{x_name} Count',
            yaxis2_title=f'{y_name} (Mean)',
            legend_title="Legend",
            template="plotly_white",
            showlegend=True
        )
        
        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', secondary_y=False)
        fig.update_yaxes(showgrid=False, secondary_y=True)  # Only grid for the first y-axis
    
        # Show the interactive plot
        return fig.to_html(full_html=False)

class TimeSeriesPlotter(metaclass=PlotMeta):
    name = "time series"
    
    @staticmethod
    def plot(df, date_col, y_col):
        """
        Plots a time series chart with different period groupings (day, month, quarter, year).
        
        Parameters:
        - df: The DataFrame containing the data.
        - date_col: The name of the column containing the date values.
        - y_col: The name of the column containing the values to plot.
        """
        if (date_col == y_col):
            return get_error_html("X shouldn't be the same as Y!")
        
        # group_method = 'mean'
        try:
            new_col = pd.to_datetime(df[date_col])
        except:
            return get_error_html(f"{date_col} cannot be converted to date format.")
        
        new_col_name = date_col
        chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        while new_col_name in df:
            new_col_name = date_col + '_' + '.'.join(list(np.random.choice(chars, 6)))
        
        df[new_col_name] = new_col
        
        df_date = (df.groupby([new_col_name]).agg({y_col: 'sum', date_col: 'count'})
                   .reset_index().rename(columns={y_col: 'sum', date_col: 'count',
                                                  new_col_name: date_col})
                  )
        df_date['mean'] = df_date['sum'] / df_date['count']
        
        
        # Create a dictionary for period formatting and labels
        periods = {
            'month': 'M',         # Month period ('M' for month)
            'quarter': 'Q',      # Quarter period ('Q' for quarter)
            'year': 'Y'        # Year period ('Y' for year)
        }
    
        # Loop through the desired periods and perform the same operations
        period_data = {date_col: df_date}
        
        for period_name, period_code in periods.items():
            # Copy the original df_date
            df_period = df_date.copy()
            
            # Create a period-based column
            if period_name == 'month':
                df_period[period_name] = df_period[date_col].dt.to_period(period_code).dt.strftime('%Y %b')  # Custom format for month
            elif period_name == 'quarter':
                df_period[period_name] = df_period[date_col].dt.to_period(period_code).dt.strftime('%Y Q%q')  # Custom format for quarter (e.g., Q1 2023)
            else:
                df_period[period_name] = df_period[date_col].dt.to_period(period_code).dt.strftime('%Y')      # Custom format for year
                
            # Group by the new period and aggregate sales data
            df_period = df_period.groupby(period_name).agg({'sum': 'sum', 'count': 'sum'}).reset_index()
            
            # Calculate average y_col
            df_period['mean'] = df_period['sum'] / df_period['count']
            
            # Store the result
            period_data[period_name] = df_period
            
        # Create figure with default (daily) data
        fig = go.Figure()
        
        # Add trace for daily data
        fig.add_trace(go.Scatter(x=df_date[date_col], y=df_date['mean'], mode='lines', name='Daily Sales'))
    
        y_title = col_name_to_title(y_col)
        # Generate buttons dynamically based on the period_data dictionary
        
        period_buttons = [
            {
                'label': period + ' ' + group_method,
                'method': 'update',
                'args': [
                    {'x': [data[period]], 'y': [data[group_method]]},  # x is the period (e.g., month, year), y is sales
                    {'xaxis': {'title': period}, 'yaxis': {'title': y_title}}  # Update axis labels accordingly
                ]
            }
            for period, data in period_data.items()
            for group_method in ['sum', 'mean', 'count']
        ]
    
        
        # Update the layout with dynamically generated buttons
        fig.update_layout(
            title=f"{y_title} Over Time",
            xaxis_title=col_name_to_title(date_col),
            yaxis_title=y_title,
            updatemenus=[
            {
                'buttons': period_buttons,
                'direction': 'down',
                'showactive': True
            }
        ]
        )

        df.drop([new_col_name], axis=1, inplace=True)
        
        return fig.to_html(full_html=False)


class ScatterPlotter(metaclass=PlotMeta):
    name = 'scatter plot'
    @staticmethod
    def plot(df: pd.DataFrame, x_col: str, y_col: str) -> None:
        """
        Create an interactive scatter plot where the size of the points is proportional to the count of occurrences.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The name x column in dataframe. X values to plot.
        y_col (str or list): String or list of strings. One or more columns from a pandas DataFrame to plot against x.
        
        Returns:
        None: Displays the scatter plot.
        """
        if (x_col == y_col):
            return get_error_html("X shouldn't be the same as Y!")
            
        if type(y_col) == str:
            y_col = [y_col]
        grouped_data = group_by_count(pd.concat([df[x_col], df[y_col]], axis=1),
                                      x_col, y_col)
    
        fig = px.scatter(title="Interactive Scatter Plot with Point Size Based on Count")
    
        colors = px.colors.qualitative.Plotly
        x_name = col_name_to_title(x_col)
        y_name = ', '.join([col_name_to_title(y) for y in y_col])
        for index, (y, group) in enumerate(grouped_data.items()):
            scatter_fig = px.scatter(
                data_frame=group,
                x=x_col,
                y=y,
                size='count',
                hover_name=y,
                color_discrete_sequence=[colors[index % len(colors)]],
                title=y
            )
    
            fig.add_traces(scatter_fig.data)
    
        fig.update_layout(
            xaxis_title=x_name,
            yaxis_title=y_name,
            legend_title_text=y_name,
            title_text=f"Scatter Plot with Point Size Based on Count of {y_name}  and {x_name}"
        )
    
        return fig.to_html(full_html=False)
    
def col_name_to_title(column_name: str) -> str:
    '''
    Changes '_' to ' ' and capitalizes each word in the column.
    '''
    return ' '.join(column_name.split('_')).title()

class WhiskerPlot(metaclass=PlotMeta):
    name = 'whisker plot'
    @staticmethod
    def plot(df: pd.DataFrame, x_col: str, y_col: str) -> None:
        """
        Create an interactive box plot showing the variation of y_col across different values of x_col.
    
        Parameters:
        df (pd.DataFrame): The DataFrame containing x_col and y_col.
        x_col (str): A name of the column for Ox.
        y_col (str): A name of the column for Oy.
    
        Returns:
        None: Displays the interactive box plot.
        """
        if (x_col == y_col):
            return get_error_html("X shouldn't be the same as Y!")
            
        # Check if the required columns exist
        if y_col not in df.columns or x_col not in df.columns:
            raise ValueError(f"Columns '{y_col}' or '{x_col}' not found in the DataFrame.")
        
        # Create the box plot
        x_name = col_name_to_title(x_col)
        y_name = col_name_to_title(y_col)
        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            title=f"{y_name} Distribution Across {x_name}",
            labels={y_col: y_name, x_col: x_name}
        )
        
        return fig.to_html(full_html=False)
    
class DistributionPlot(metaclass=PlotMeta):
    name = 'distribution'
    @staticmethod
    def plot(df: pd.DataFrame, column: str, y_column: str='', bins: int=30) -> None:
        """
        Plot the interactive distribution of a column using Plotly.
    
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column.
        bins (int): Number of bins in the histogram.
    
        Returns:
        None: Displays an interactive histogram of the column distribution.
        """
        # Check if the specified column exists
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
        # Drop any missing values in the column
        if df[column].isnull().any():
            w = f'Warning: Column {column} contains {df[column].isnull().sum()} null values. These will be ignored in the plot.'
            warnings.warn(w, UserWarning)
        
        column_upd = df[column].dropna()
        
        # Create the interactive histogram using Plotly
        fig = px.histogram(df, x=column, nbins=bins, title=f'{column.title()} Distribution')
        fig.update_traces(marker_line_color='darkblue', marker_line_width=1)
    
        # Customize the plot
        fig.update_layout(
            xaxis_title=column.title(),
            yaxis_title='Frequency',
            title_font_size=16
        )
        
        # Show the plot with hover information (Plotly automatically provides hover info)
        return fig.to_html(full_html=False)
    

class DensityHeatmap(metaclass=PlotMeta):
    name = 'density heatmap'
    @staticmethod
    def plot(df: pd.DataFrame, x_col: str, y_col: str) -> None:
        """
        Generates a density heatmap to visualize the relationship between two categorical or continuous variables,
        displaying the frequency (count) of occurrences for each unique combination of values.
    
        Parameters:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        x_col (str): The name of the column to be used as the x-axis (e.g., 'enrollment_season').
        y_col (str): The name of the column to be used as the y-axis (e.g., 'internet_hours_per_day').
        """
        if (x_col == y_col):
            return get_error_html("X shouldn't be the same as Y!")
            
        df_grouped = df.groupby([x_col, y_col]).size().reset_index(name='count')
        x_name = col_name_to_title(x_col)
        y_name = col_name_to_title(y_col)
        fig = px.density_heatmap(df_grouped, x=x_col, y=y_col, z='count',
                                 title=f"Heatmap of {y_name} Per Day by {x_name}",
                                 labels={x_name: x_col,
                                         y_name: y_col},
                                nbinsx=len(df[x_col].unique()),
                                 nbinsy=len(df[y_col].unique())
                                 )
        return fig.to_html(full_html=False)
        