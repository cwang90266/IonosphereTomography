#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:19:57 2026

@author: austinhunter
"""
import requests
import pandas as pd
import io

def fetch_giro_data():
    url = "https://giro.uml.edu/didbase/scaled.php"

    # 1. Mask the script as a standard web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    # 2. The expanded payload matching DIDBase PHP variables
    # Note: PHP often uses separate keys for start and end dates
    payload = {
        'ursi': 'BC840',
        'year1': '2026',
        'month1': '03',
        'day1': '01',
        'year2': '2026',
        'month2': '03',
        'day2': '31',
        # For multiple characteristics, PHP forms often require list format
        'chars[]': ['foF2', 'hmF2'], 
        'format': 'txt',     # Or try 'ascii' if 'txt' still fails
        'submit': 'Search'   # The crucial form trigger
    }

    print(f"Requesting data from {url}...")
    
    # Pass the headers along with the payload
    response = requests.post(url, data=payload, headers=headers)

    if response.status_code == 200:
        # Check if the response is still HTML
        if response.text.strip().startswith('<!DOCTYPE html') or response.text.strip().startswith('<html'):
            print("Error: The server still returned an HTML webpage. The form keys need exact verification.")
            return None
            
        print("Data retrieved successfully. Parsing...")
        
        try:
            # Use sep=r'\s+' to handle irregular spacing and fix the Pandas warning
            df = pd.read_csv(
                io.StringIO(response.text), 
                sep=r'\s+', 
                comment='#'
            )
            return df
            
        except Exception as e:
            print(f"Parsing error: {e}")
            return None
            
    else:
        print(f"Connection failed. HTTP Status: {response.status_code}")
        return None

# Execute
giro_df = fetch_giro_data()

if giro_df is not None:
    print(giro_df.head())