# test window_edps content
import pickle

edps_path = '/home/pin/Desktop/tomography_project/Data/WINDOW_EDPS/2025-11-18_1020_binall_window_edps.pkl'
# 
# {
#     'time',
#     'lat',
#     'lon',
#     'alt_site_km',
#     'alt_km',
#     'ne_m3',
#     'dne_m3',
#     'ti_K',
#     'tr',
#     'source_file',
#     'kindat'
# }
# 
with open(edps_path, "rb") as f:
    window_edps = pickle.load(f)

    print(type(window_edps))
    print(len(window_edps))
    print(type(window_edps[0]))
    print(window_edps[0].keys())