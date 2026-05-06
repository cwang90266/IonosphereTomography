import requests
import pandas as pd
from io import StringIO

def fetch_and_parse_data(url):
    """
    Fetch a text file from a website, parse header and data.

    Args:
        url (str): The URL to the text file

    Returns:
        pandas.DataFrame: Data parsed into a DataFrame
    """
    response = requests.get(url)
    response.raise_for_status()
    lines = response.text.splitlines()

    header_lines = []
    for i, line in enumerate(lines):
        if line.startswith('#'):
            header_lines.append(line)
        else:
            data_start = i
            break
    else:
        data_start = len(lines)

    # Column headings are in last header line (strip # and whitespace)
    columns = header_lines[-1].lstrip('#').strip().split()
    data_rows = [line.strip().split() for line in lines[data_start:] if line.strip()]

    df = pd.DataFrame(data_rows, columns=columns)
    # Optional: try converting numeric columns to appropriate types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    return df

# Example usage:
# url = 'https://example.com/data.txt'
# df = fetch_and_parse_data(url)