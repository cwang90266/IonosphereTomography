#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 08:23:06 2026

@author: cwang
"""
import sys
import os
import numpy as np

target_dir = os.path.abspath("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography/IRI_Sample_Inputs")
sys.path.insert(0, target_dir)
target_dir = os.path.abspath("/Users/cwang/Documents/Consulting/PlanetIQ/Code/IonosphereTomography/EDPSamples")
sys.path.insert(0, target_dir)
from IRI_Sample_inputs import IRI_Sample_Inputs as IRIs
import edp_samples as EDPS_Util
from  edp_samples import EDPSamples as EDPS

DateTime_str = "2003-11-21T12"
Sample_Param = IRIs(DateTime_str)

hour_sample_range = 6
ap_sample_range = 31
f107_sample_range = 31
ig_sample_range = 48
rz_sample_range = 48
Sample_Param = Sample_Param.quantileSamples(
                   hour_sample_range,
                   ap_sample_range,
                   f107_sample_range,
                   ig_sample_range,
                   rz_sample_range)

altitude = np.arange(100,1000,10)

EDPSam = EDPS(DateTime_str,"Point", altitude,Sample_Param,evaluate_iri=1,Lon = 20, Lat= 30)

EDPSam.saveNetCDF("test.nc")

EDPSam_1 = EDPS.fromNetCDF("test.nc")