import pandas as pd

def from_CPT_to_df(data_string):

    
    # Split the string into lines
    lines = data_string.strip().split('\n')

    # Remove the horizontal line separators
    lines = [line for line in lines if not line.startswith('+')]

    # Find the index where the column names are defined
    header_separator = 3

    # Extract the column names
    columns = [column.strip() for column in lines[header_separator-1].split('|') if column.strip()]

    # Find the index where the data rows start
    data_start = header_separator

    # Extract the data rows
    data_rows = [line.split('|')[1:-1] for line in lines[data_start:]]

    # Clean up and format the data
    data = []
    for row in data_rows:
        clean_row = [item.strip() for item in row]
        data.append(clean_row)

    df = pd.DataFrame(data, columns=columns)
    return df