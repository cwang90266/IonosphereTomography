#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 08:23:06 2026

@author: cwang
"""
import sys
import os
import numpy as np
 
Source_ROOT = os.getenv("Tomography_Source_Folder")
Run_Folder = os.getenv("Tomography_Run_Folder")

target_dir = os.path.abspath(Source_ROOT+"/IRI_Sample_Inputs")
sys.path.insert(0, target_dir)
target_dir = os.path.abspath(Source_ROOT+"/EDPSamples")
sys.path.insert(0, target_dir)

from IRI_Sample_inputs import IRI_Sample_Inputs as IRIs
from edp_samples import EDPSamples as EDPS
from edp_samples import read_iri_namelist_log as readlog

DateTime_str = "2003-11-21T12"
Sample_Param_Database = IRIs(DateTime_str)
Sample_Param_Database.save_to_file("IRI_Param_Database")

Sample_Param_Database=IRIs.fromPickle(filename='IRI_Param_Database')


hour_sample_range = 6
ap_sample_range = 31
f107_sample_range = 31
ig_sample_range = 48
rz_sample_range = 48
Sample_Param = Sample_Param_Database.quantileSamples(
                   hour_sample_range,
                   ap_sample_range,
                   f107_sample_range,
                   ig_sample_range,
                   rz_sample_range)
nSample = 2000
Sample_Param = Sample_Param_Database.randomSamples(
                   hour_sample_range,
                   ap_sample_range,
                   f107_sample_range,
                   ig_sample_range,
                   rz_sample_range,
                   nSample)
altitude = np.arange(100,1000,10)

EDPSam_Point = EDPS(DateTime_str,"Point", altitude,Sample_Param,evaluate_iri=1,Lon = 20, Lat= 30)
EDPSam_Point.saveNetCDF(Run_Folder+"/EDPSam_Point.nc")
idx_nan = readlog(Run_Folder+"/iri2020_namelist_driver.log")
idx_nan_sample = idx_nan[:,0]
idx_nan_sample = idx_nan_sample.tolist()
Sample_Param_nan = Sample_Param.iloc[idx_nan_sample]

EDPSam_MidLat = EDPS(DateTime_str,"Rectangle", altitude,Sample_Param,evaluate_iri=1,
                     minLon = -20, maxLon=20, dLon=2.5, minLat= -30,maxLat=30, dLat=2.5)
EDPSam_MidLat.saveNetCDF("EDPSam_MidLat.nc")

EDPSam_Polar = EDPS(DateTime_str,"Polar", altitude,Sample_Param,evaluate_iri=1,
                    minLat= 75, dLat=2.5)
EDPSam_Polar.saveNetCDF("EDPSam_Polar.nc")
