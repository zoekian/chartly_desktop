import pandas as pd
import numpy as np
from math import floor, ceil

def autoround(col: pd.Series) -> pd.Series:
    lower = np.percentile(col, 10)
    upper = np.percentile(col, 90)
    if upper == lower:
        return col
    return np.round(col, 1-floor(np.log10(upper - lower)))


def clip_outliers_iqr(column: pd.Series) -> pd.Series:
    """
    Clip outliers in a DataFrame column based on the IQR method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column for which to clip outliers.

    Returns:
    pd.Series: A Series with outliers clipped based on the IQR method.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Define the lower and upper whiskers (thresholds)
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    
    # Clip the values in the column to the whiskers
    if column.dtype == int:
        lower_whisker = floor(lower_whisker)
        upper_whisker = ceil(upper_whisker)
        
    clipped_column = np.clip(column, lower_whisker, upper_whisker)
    
    return clipped_column

def get_columns_to_clip(df: pd.DataFrame) -> list:
    '''
    Get all numerical columns of df that possibly should be clipped.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    List of columns that possibly should be clipped.
    '''
    to_clip = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].nunique() > 10:
                to_clip.append(column)
    return to_clip

def determine_rounding(df: pd.DataFrame) -> dict:
    """
    Analyze each column in the DataFrame to determine if it should be rounded and at what precision.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    dict: A dictionary with column names as keys and rounding precision as values.
    """
    rounding_info = {}  # Dictionary to store rounding information for each column
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if pd.api.types.is_integer_dtype(df[column]):
                rounding_info[column] = 0  # Integers do not need rounding
            elif pd.api.types.is_float_dtype(df[column]):
                # Check for the precision of the numbers
                min_value = df[column].min()
                max_value = df[column].max()
                unique_values = df[column].nunique()

                # Determine rounding precision based on value range and unique values
                if unique_values > 10:  # More than 10 unique values suggest more precision
                    if min_value >= 0 and max_value < 1:
                        rounding_info[column] = 4  # Precision of 4 decimal places for [0, 1)
                    elif min_value >= 0 and max_value < 100:
                        rounding_info[column] = 2  # Precision of 2 decimal places for [0, 100)
                    elif max_value >= 100:
                        # For large values, round to the nearest significant figure
                        rounding_info[column] = 'significant'  # Indicating to use significant rounding
                    else:
                        rounding_info[column] = 2  # Default to 2 decimal places for mixed range
                else:
                    rounding_info[column] = 2  # Default precision for fewer unique values

    return rounding_info  # Return the dictionary with rounding information

# Function to round large numbers to significant figures
def round_large_numbers(df, column, significant_digits=3) -> pd.Series:
    """
    Round a number to the nearest significant figures.

    Parameters:
    df (pd.DataFrame): DataFrame that contains the column to round.
    column (str): The column name which should be rounded.
    significant_digits (int): The number of significant digits.

    Returns:
    pd.Series: The rounded column.
    """
    value = np.percentile(np.abs(df[column][np.logical_not(df[column].isna())]), 50)
    if value == 0:
        return 0
    else:
        ndigits = significant_digits - int(np.floor(np.log10(value)))
        # Calculate the power of ten for rounding
        return np.round(df[column], ndigits)
        
def clip_and_round_all(df: pd.DataFrame) -> None:
    '''
    Clips and rounds all numerical features in df.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    '''
    for key in df:
        if df[key].nunique() > 50 and df[key].dtype != object:
            
            new_col = clip_outliers_iqr(df[key][np.logical_not(df[key].isnull())])
            to_round = floor(np.log10(np.percentile(new_col, 50))) - 1
            df[key] = np.round(new_col, to_round)
    