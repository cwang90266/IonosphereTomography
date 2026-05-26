#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:47:16 2026

@author: cwang
"""

import os

def initialize_ionosphere_tomography(Source_Root: str,
                                     Run_Folder:str,
                                     RO_Data_Folder:str):
    os.environ["Tomography_Source_Folder"] = Source_Root
    os.environ["IRI_Root"] = Source_Root + "/iri2020_new/src/iri2020"
    os.environ["Tomography_Run_Folder"] = Run_Folder
    os.environ["RO_Folder"] = RO_Data_Folder
    