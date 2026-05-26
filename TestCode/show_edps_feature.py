#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Plot the distribution of physical input parameters and the key 
   charateristics of the resulting electron density profiles.
   
Created on Fri May 22 12:58:06 2026

@author: Chunming Wang
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show_edps_feature(iri_parameters: pd.DataFrame, feature_edps:np.array):
    layout =[
        ["A","B","C","D"],
        ["E","E","F","F"]
        ]
    fig, ax = plt.subplot_mosaic(layout, figsize=(12, 6))
    ax["A"].set_title("F10.7")
    ax["B"].set_title("ap")
    ax["C"].set_title("IG12")
    ax["D"].set_title("Rz12")
    ax["E"].set_title("Densities at Key Heights")
    ax["F"].set_title("Key Heights")
    
    fig.suptitle('Distribution of Input Parameters and Key EDP Characteristics', fontsize=16)
    fig.show()