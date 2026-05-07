#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:48:41 2026

@author: cwang
"""
from matplotlib.pyplot import figure
import datetime
import matplotlib.dates as mdates
import requests
import urllib.request
import numpy as np
import pandas as pd
from dateutil.parser import parse

def get_apf107():
    """
    Get updated data file apf107.dat

    Parameters
    ----------
    None
    
    Returns apf107
    -------
 
    """
    url = "https://chain-new.chain-project.net/echaim_downloads/apf107.dat"
    response = requests.get(url)
    response.raise_for_status()
    input_lines = response.text.splitlines()
    
    local_file = "apf107.dat"
    urllib.request.urlretrieve(url, local_file)

    apf107={
        "yr":[],
        "mn":[],
        "dy":[],
        "iapda":[],
        "iiap":[],
        "ir":[],
        "f107":[],
        "f107_81":[],
        "f107_365":[]
            }
    for line in input_lines:
        # Process each line.
        yy = int(line[0:3])
        if yy < 30:
            yy = 2000+yy
        else:
            yy = 1900+yy

        apf107["yr"].append(yy)
        apf107["mn"].append(int(line[3:6]))
        apf107["dy"].append(int(line[6:9]))
        apf107["iapda"].append(int(line[9+0:9+3]))
        apf107["iiap"].append([int(line[12+i*3:15+i*3]) for i in range(8)])
        apf107["ir"].append(int(line[36:39]))
        apf107["f107"].append(float(line[39:44]))
        apf107["f107_81"].append(float(line[44:49]))
        apf107["f107_365"].append(float(line[49:54]))

    return apf107

def get_ig_rz():
    """
    Get updated data file ig_rz.dat

    Parameters
    ----------
    None

    Returns
    -------
    data_record : dict
        Dictionary containing parsed IG/Rz data.
    """

    revision_date = []
    start_end_date = []
    ig = []
    rz = []
    url = "https://chain-new.chain-project.net/echaim_downloads/ig_rz.dat"
    response = requests.get(url)
    response.raise_for_status()
    input_lines = response.text.splitlines()
    
    local_file = "ig_rz.dat"    
    urllib.request.urlretrieve(url, local_file)

    # 0: revision date, 1: start/end month/year, 2: ig, 3: rz
    lines = [line.strip() for line in input_lines if line.strip() != ""]

    if len(lines) < 6:
        raise ValueError("Input file seems too short or incorrectly formatted.")

    # 0: revision date, 2: start/end, 4: IG start, IG data..., then a blank, then Rz
    revision_date = [int(x) for x in lines[0].split(",") if x.strip().isdigit()]
    start_end_date = [int(x) for x in lines[1].split(",") if x.strip().isdigit()]

    # The IG data spans from line 4 until the next blank, then Rz starts after that
    # Find the separation between IG and Rz
    ig_start = 2
    # Find where the blank line is between IG and Rz (which we already stripped)
    # But we can assume IG and Rz are equal in length (as per file comments), so:
    n_header_lines = 2  # revision, blank, start_end, blank
    n_ig_rz = (len(lines) - n_header_lines) // 2
    ig_lines = lines[ig_start : ig_start + n_ig_rz]
    rz_lines = lines[ig_start + n_ig_rz : ig_start + 2 * n_ig_rz]

    for line in ig_lines:
        for val in line.split(","):
            val = val.strip()
            if val:
                try:
                    ig.append(float(val))
                except ValueError:
                    continue  # Skip non-numeric

    for line in rz_lines:
        for val in line.split(","):
            val = val.strip()
            if val:
                try:
                    rz.append(float(val))
                except ValueError:
                    continue

    data_record = {
        "Revision": revision_date,
        "Start_end_month": start_end_date,
        "ig": ig,
        "rz": rz
    }
    return data_record

def show_iri_inputs(apf107,ig_rz):
    """
    Visualize time history of input parameters ap, f107, sun spot, ig and rz
    
    Parameter:
        apf107  : Dictionary, output of read_apf107
        ig_rz   : Dictionary, output of read_ig_rz
    """
    fig = figure(figsize=(10, 10))
    axs = fig.subplots(4, 1)
    fig.suptitle("Time History of Input Parameters to IRI2020")

    datenum = [datetime.date(int(yy), int(mm), int(dd)) for yy, mm, dd in 
            zip(apf107["yr"], apf107["mn"], apf107["dy"])]
    
    pn = axs[0]
    pn.plot(datenum, apf107["iapda"], marker='o', linestyle='-')
    pn.plot(datenum, apf107["iiap"], marker='o', linestyle='-')
    fig.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #fig.gca().autofmt_xdate()  # Auto rotate date labels
    pn.set_xlabel('Date')
    pn.set_ylabel('AP')

    # pn = axs[1]
    # pn.plot(datenum, apf107["ir"], marker='o', linestyle='-')
    # fig.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    # fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #fig.gca().autofmt_xdate()  # Auto rotate date labels
    #pn.set_xlabel('Date')
    #pn.set_ylabel('Sun Spot')
    
    pn = axs[1]
    pn.plot(datenum, apf107["f107"], marker='o', linestyle='-')
    pn.plot(datenum, apf107["f107_81"], marker='o', linestyle='-')
    pn.plot(datenum, apf107["f107_365"], marker='o', linestyle='-')
    fig.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #fig.gca().autofmt_xdate()  # Auto rotate date labels
    pn.set_xlabel('Date')
    pn.set_ylabel('F10.7')

    start_end_month=ig_rz["Start_end_month"]
    if start_end_month[0] == 1 :
        first_month =12
        first_year = start_end_month[1] -1
    else:
        first_month = start_end_month[0]-1
        first_year = start_end_month[1]

    if start_end_month[2] == 12 :
        last_month =1
        last_year = start_end_month[3] + 1
    else:
        last_month = start_end_month[2]+1
        last_year = start_end_month[3]



    # Construct vectors 'month' and 'year' spanning from the first month/year to the last month/year
    month = []
    year = []
    y, m = first_year, first_month
    while (y < last_year) or (y == last_year and m <= last_month):
        month.append(m)
        year.append(y)
        m += 1
        if m > 12:
            m = 1
            y += 1
    day = [15] * len(year)
    datenum = [datetime.date(int(yy), int(mm), int(dd)) for yy, mm, dd in 
            zip(year, month, day)]

    pn = axs[2]
    pn.plot(datenum, ig_rz["ig"], marker='o', linestyle='-')
    fig.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #fig.gca().autofmt_xdate()  # Auto rotate date labels
    pn.set_xlabel('Date')
    pn.set_ylabel('ig')

    pn = axs[3]
    pn.plot(datenum, ig_rz["rz"], marker='o', linestyle='-')
    fig.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #fig.gca().autofmt_xdate()  # Auto rotate date labels
    pn.set_xlabel('Date')
    pn.set_ylabel('Rz')

    fig.show()
    
class IRI_Sample_Inputs:
    def __init__(self, DateTime_str: str):
        DateTime_int = parse(DateTime_str)
        self.year = DateTime_int.year
        self.month = DateTime_int.month
        self.day = DateTime_int.day
        if hasattr(DateTime_int,'hour'):
            self.hour = DateTime_int.hour
        else:
            self.hour = 0
            
        if hasattr(DateTime_int,'minute'):
            self.minute = DateTime_int.minute
        else: 
            self.minute = 0
            
        if hasattr(DateTime_int,"second"):
            self.second=DateTime_int.second 
        else:
            self.second = 0
            
        self.apf107 = get_apf107()
        self.ig_rz = get_ig_rz()
        
        # Identify the indice of the current time in the array apf107 and ig_rz
        datenum_f107 = [datetime.date(int(yy), int(mm), int(dd)) for yy, mm, dd in 
            zip(self.apf107["yr"], self.apf107["mn"], self.apf107["dy"])]
        datenum_sim=datetime.date(self.year,self.month,self.day)
        self.current_idx_f107=datenum_f107.index(datenum_sim)
        #
        start_end_month=self.ig_rz["Start_end_month"]
        if start_end_month[0] == 1 :
            first_month =12
            first_year = start_end_month[1] -1
        else:
            first_month = start_end_month[0]-1
            first_year = start_end_month[1]
    
        if start_end_month[2] == 12 :
            last_month =1
            last_year = start_end_month[3] + 1
        else:
            last_month = start_end_month[2]+1
            last_year = start_end_month[3]
 
        month = []
        year = []
        y, m = first_year, first_month
        while (y < last_year) or (y == last_year and m <= last_month):
            month.append(m)
            year.append(y)
            m += 1
            if m > 12:
                m = 1
                y += 1
        day = [15] * len(year)
        datenum_igrz = [datetime.date(int(yy), int(mm), int(dd)) for yy, mm, dd in 
            zip(year, month, day)]
        datenum_sim=datetime.date(self.year,self.month,15)
        self.current_idx_igrz=datenum_igrz.index(datenum_sim)

    def quantileSamples(self,
                       hour_sample_range:int = None,
                       ap_sample_range: int = None,
                       f107_sample_range: int = None,
                       ig_sample_range: int = None,
                       rz_sample_range: int = None):
        
        # Define range of parameters
        if hour_sample_range == None:
            hour_range=[None]
        else:
            hour_range=[24+i if i<0 else i for i in range(self.hour-hour_sample_range,
                                         self.hour+hour_sample_range+1)]
            hour_range=[i-24 if i>24 else i for i in hour_range]
            hour_range.append(None)
            
        f107_range=self.apf107["f107"]
        if f107_sample_range == None:
            f107_range=[None]
        else:
            idx_start=max((0,self.current_idx_f107-f107_sample_range))
            idx_end=min((self.current_idx_f107+f107_sample_range,len(f107_range)-1))
            f107_range=f107_range[idx_start:idx_end]
            f107_range=[min(f107_range),float(np.mean(f107_range)),max(f107_range)]
            f107_range.append(None)
        #
        ap_range=self.apf107["iiap"]
        if ap_sample_range == None:
            ap_range=[None]
        else:                
            idx_start=max((0,self.current_idx_f107-ap_sample_range))
            idx_end=min((self.current_idx_f107+ap_sample_range,len(ap_range)-1))
            ap_range=ap_range[idx_start:idx_end]
            ap_range=[min(min(ap_range)),float(np.mean(ap_range)),
                      
                      max(max(ap_range))]
            ap_range.append(None)
        #
        if ig_sample_range == None:
            ig12_range=[None]
        else:
            ig12_range=self.ig_rz["ig"]
            idx_start=max((0,self.current_idx_igrz-ig_sample_range))
            idx_end=min((self.current_idx_igrz+ig_sample_range,len(ig12_range)-1))
            ig12_range=ig12_range[idx_start:idx_end]
            f107_range=[min(ig12_range),float(np.mean(ig12_range)),max(ig12_range)]
            ig12_range.append(None)
        #
        if rz_sample_range == None:
            Rz12_range=[None]
        else:
            Rz12_range=self.ig_rz["rz"]
            idx_start=max((0,self.current_idx_igrz-rz_sample_range))
            idx_end=min((self.current_idx_igrz+rz_sample_range,len(Rz12_range)-1))
            Rz12_range=Rz12_range[idx_start:idx_end]
            Rz12_range=[min(Rz12_range),float(np.mean(Rz12_range)),max(Rz12_range)]
            Rz12_range.append(None)
        
        Samples={'hour':[], 'f107':[],'ap':[],'ig12':[],'rz12':[]}
        for hour in hour_range:
            for f107 in f107_range:
                for ap in ap_range:
                    for ig12 in f107_range:
                        for Rz12 in Rz12_range:
                            Samples['hour'].append(hour)
                            Samples['f107'].append(f107)
                            Samples['ap'].append(ap)
                            Samples['ig12'].append(ig12)
                            Samples['rz12'].append(Rz12)
        
        return pd.DataFrame(Samples)
            
         
