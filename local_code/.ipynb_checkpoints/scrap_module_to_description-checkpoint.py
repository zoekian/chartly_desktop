import warnings
import pandas as pd

def add_text_to_collection(vectorstore,
                       text: str, prefix: str, separator: str ='\n', chunk_size: int = 5000,
                       max_size: int = 7000, metadata: dict = {},
                      replace: bool = True, shift_range: int = 0
                      ) -> None:
    '''
This function splits a given text into smaller chunks based on the specified separator. Each chunk is labeled with an identifier that uses a prefix.It is useful when handling large text data that needs to be processed in smaller, more manageable segments.

Parameters:
text (str): The input text string that needs to be split into smaller chunks.
prefix (str): A string prefix that will be used to generate unique identifiers for each chunk.
separator (str): The character or string used to split the text. For instance, this could be a newline (\n\n) to split text by paragraphs.
chunk_size (int, optional, default=5000): The size of each chunk in terms of the number of characters. If the text exceeds the maximum size, it is split into chunks of this size.
max_size (int, optional, default=7000): The maximum allowable size for the text before it is split. Texts below this size are not split.

Returns:
tuple: A tuple containing:
A list of text chunks.
A corresponding list of unique identifiers generated using the provided prefix.

Behavior:
If the length of the text exceeds the max_size parameter, the function uses the provided separator to split the text into chunks of size chunk_size. Otherwise, it treats the text as a single chunk.
It assigns unique IDs to each chunk with a specific prefix and a number suffix (e.g., 'id_pf0', 'id_pf1').
    '''
    if len(text) > max_size:
        splitter = CharacterTextSplitter(separator = "\n\n",
                                        chunk_size=chunk_size, 
                                        chunk_overlap=0,
                                        length_function=len)

        texts = splitter.split_text(text)
    else:
        texts = [text]
    
    if not replace:
        ids = vectorstore.get()['ids']
        shift_range = max([int(x[len(prefix):]) 
                           for x in list(
                               filter(lambda x: (x.startswith(prefix)), ids)
                           )]) + 1
    ids = []
    # pf - for plot functions
    for i in range(shift_range, len(texts) + shift_range):
        ids.append(prefix + str(i))
        
    vectorstore.add_texts(texts, ids=ids, metadatas=[metadata] * len(ids))


def text_to_ingestible(text: str, prefix: str, separator: str, chunk_size: int = 5000,
                       max_size: int = 7000) -> tuple:
    '''
This function splits a given text into smaller chunks based on the specified separator. Each chunk is labeled with an identifier that uses a prefix.It is useful when handling large text data that needs to be processed in smaller, more manageable segments.

Parameters:
text (str): The input text string that needs to be split into smaller chunks.
prefix (str): A string prefix that will be used to generate unique identifiers for each chunk.
separator (str): The character or string used to split the text. For instance, this could be a newline (\n\n) to split text by paragraphs.
chunk_size (int, optional, default=5000): The size of each chunk in terms of the number of characters. If the text exceeds the maximum size, it is split into chunks of this size.
max_size (int, optional, default=7000): The maximum allowable size for the text before it is split. Texts below this size are not split.

Returns:
tuple: A tuple containing:
A list of text chunks.
A corresponding list of unique identifiers generated using the provided prefix.

Behavior:
If the length of the text exceeds the max_size parameter, the function uses the provided separator to split the text into chunks of size chunk_size. Otherwise, it treats the text as a single chunk.
It assigns unique IDs to each chunk with a specific prefix and a number suffix (e.g., 'id_pf0', 'id_pf1').
    '''
    if len(text) > max_size:
        splitter = CharacterTextSplitter(separator = "\n\n",
                                        chunk_size=chunk_size, 
                                        chunk_overlap=0,
                                        length_function=len)

        text = splitter.split_text(text)
    else:
        text = [text]
    ids = []
    # pf - for plot functions
    for i in range(len(text)):
        ids.append(prefix + str(i))
    return text, ids

import pandas as pd
import numpy as np
import warnings

def generate_func_descr_df(code: str) -> pd.DataFrame:
    """
    Generates a DataFrame summarizing Python functions by extracting function names, descriptions, inputs, and outputs from the code and docstrings.

    Parameters:
    code (str): A multiline string containing Python function definitions.

    Returns:
    pd.DataFrame: A DataFrame containing summaries of all functions found in the code.
    """
    func_descr = {'function_name': [], 'description': [], 'input': [], 'output': []}

    docstring_started = False
    docstring_content = []
    param_lines = []
    return_lines = []
    
    param_section_started = False
    return_section_started = False

    for line in code.splitlines():
        # Detect function definition
        if line.strip().startswith('def '):
            # Extract function name and parameters
            func_descr['function_name'].append(line.split(' ')[1].split('(')[0])
            func_descr['description'].append('')  # Placeholder for description
            func_descr['input'].append(line.split('(')[1].split(')')[0])
            if '->' in line:
                func_descr['output'].append(line.split('->')[-1].strip().rstrip(':'))
            else:
                warnings.warn(f"Missing return info in function: {func_descr['function_name'][-1]}")
                func_descr['output'].append('Unknown')
            
        # Detect start of docstring (""" or ''')
        if line.strip().startswith(('"""', "'''")):
            docstring_started = not docstring_started 
        elif docstring_started:
            # Look for Parameters section in docstring
            if 'Parameters:' in line:
                param_section_started = True

            # Look for Returns section in docstring
            if 'Returns:' in line:
                return_section_started = True
                param_section_started = False
                
            if not return_section_started and not param_section_started:
                func_descr['description'][-1] += ' ' + line.strip()
        
            # Capture parameters
            if param_section_started:
                if ':' in line:
                    param_lines.append(line.strip())

            # Capture return information
            if return_section_started:
                return_lines.append(line.strip())
       
        # If parameters were found in the docstring, replace basic input with this
        if not docstring_started:
            if param_lines:
                func_descr['input'][-1] = ', '.join(param_lines)

            if return_lines:
                func_descr['output'][-1] = ' '.join(return_lines)[len('Returns: '):]
            
            docstring_content = []
            param_lines = []
            return_lines = []
            param_section_started = False
            return_section_started = False

    func_descr_df = pd.DataFrame(func_descr)
    
    for key in func_descr_df:
        if np.any(func_descr_df[key] == ''):
            print(f'There are empty values in {key}. You can see it below.')
            display(func_descr_df[func_descr_df[key] == ''])
            
    
    return func_descr_df

def summarize_functions(code: str) -> str:
    """
    Summarizes Python functions by extracting function names, descriptions, inputs, and outputs from the code and docstrings.

    Parameters:
    code (str): A multiline string containing Python function definitions.

    Returns:
    str: A string summarizing all functions found in the code.
    """
    func_descr_df = generate_func_descr_df(code)
    func_descr_df['descr'] = func_descr_df.apply(lambda x: '; '.join([i + ': ' + str(v).strip()
                                                        for i, v in zip(x.index, x.values)]),
                                   axis=1)

    return '\n\n'.join(func_descr_df['descr'])
