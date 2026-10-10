from matplotlib.pyplot import figure
import datetime
import matplotlib.dates as mdates
import numpy as np

def read_apf107(file_path):
    """
    Reads data file apf107.dat

    Parameters
    ----------
    file_path : str
        Path to folder that contains the file ig_rz.dat.

    Returns apf107
    -------
 
    """
    filename=file_path+"/apf107.dat"
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
    with open(filename, 'r') as f:
        for line in f:
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

def read_ig_rz(file_path):
    """
    Reads data file ig_rz.dat

    Parameters
    ----------
    file_path : str
        Path to folder that contains the file ig_rz.dat.

    Returns
    -------
    data_record : dict
        Dictionary containing parsed IG/Rz data.
    """
    # The main issue with the original implementation is that it processes every single token in the file inefficiently, 
    # iterating line by line and splitting by comma each time, and using a match/case that is based on the number of blank lines encountered.
    # This causes problems in parsing, as the actual structure of ig_rz.dat is multi-section and cannot be differentiated reliably merely by counting blank lines.
    # On Jupiter (Jupyter) and similar environments, this can lead to very slow, memory-inefficient, or even incorrect parsing (i.e., extremely slow definition and function execution). 
    # Instead, it's much more efficient to explicitly track which section is being read, and avoid unnecessary string operations and error-catching on every token.
    # Here's a much more efficient and robust rewrite:

    revision_date = []
    start_end_date = []
    ig = []
    rz = []
    filename = file_path + "/ig_rz.dat"

    section = 0  # 0: revision date, 1: start/end month/year, 2: ig, 3: rz
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]

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
    pn.set_ylabel('ig')

    fig.show()

    