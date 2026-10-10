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