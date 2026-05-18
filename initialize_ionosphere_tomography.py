#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:47:16 2026

@author: cwang
"""

import os

def initialize_ionosphere_tomography(IRI_Root: str, Run_Folder:str):
    os.environ["IRI_Root"] = IRI_Root
    os.environ["Tomography_Run_Folder"] = Run_Folder
    