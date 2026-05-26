#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 08:23:06 2026

@author: cwang
"""
import sys
import os
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt


Source_ROOT = os.getenv("Tomography_Source_Folder")
Run_Folder = os.getenv("Tomography_Run_Folder")

target_dir = os.path.abspath(Source_ROOT+"/IRI_Sample_Inputs")
sys.path.insert(0, target_dir)
target_dir = os.path.abspath(Source_ROOT+"/EDPSamples")
sys.path.insert(0, target_dir)

from IRI_Sample_inputs import IRI_Sample_Inputs as IRIs
from edp_samples import EDPSamples as EDPS
from edp_samples import read_iri_namelist_log as readlog

from viz_mesh_profiles import visualize_scalar_field as viz




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
EDPSam_Polar = EDPS(DateTime_str,"Polar", altitude,Sample_Param,evaluate_iri=1,
                    minLat= 75, dLat=2.5)
if "Tomography_Run_Folder" in os.environ:
        Run_Folder = os.getenv("Tomography_Run_Folder")
        EDPSam_Polar.saveNetCDF(Run_Folder+"/EDPSam_Polar.nc")
EDPSam_Polar.show_edps_feature()


idx_nan = readlog(Run_Folder+"/iri2020_namelist_driver.log")
idx_nan_sample = idx_nan[:,0]
idx_nan_sample = idx_nan_sample.tolist()
Sample_Param_nan = Sample_Param.iloc[idx_nan_sample]

DateTime_str = "2009-11-21T12"
Sample_Param_Database_1 = IRIs(DateTime_str)
nSample = 2000
Sample_Param_1 = Sample_Param_Database_1.randomSamples(
                   hour_sample_range,
                   ap_sample_range,
                   f107_sample_range,
                   ig_sample_range,
                   rz_sample_range,
                   nSample)

EDPSam_Polar_1 = EDPS(DateTime_str,"Polar", altitude,Sample_Param_1,evaluate_iri=1,
                    minLat= 75, dLat=2.5)
if "Tomography_Run_Folder" in os.environ:
        Run_Folder = os.getenv("Tomography_Run_Folder")
        EDPSam_Polar_1.saveNetCDF(Run_Folder+"/EDPSam_Polar_1.nc")

EDPSam_Polar_1.show_edps_feature()

# Compute singular value decomposition
edps = EDPSam_Polar.edps
edps_reshape = edps.reshape((edps.shape[0]*edps.shape[1], edps.shape[2]))
edps_mean = edps_reshape.mean(axis=1)
edps_mean = edps_mean.reshape((edps_reshape.shape[0], 1))
edps_0 = edps_reshape - edps_mean
edps_0s = edps_0/edps_mean
CoV_inner = edps_0s.T @ edps_0s
EVa, EVec = np.linalg.eig(CoV_inner)
EVa = EVa.reshape((edps_reshape.shape[1], 1))
EVa = np.abs(EVa)
EVa = np.sqrt(EVa)
idx = np.where(EVa>1)
idx = idx[0]
idx_max = max(idx)
#
fig = figure(figsize=(10, 10))
axs = fig.subplots(1, 1)
axs.plot(range(EVa.shape[0]), EVa, marker='o', linestyle='none')
axs.plot(np.array([idx_max, idx_max]), np.array([np.min(EVa),np.max(EVa)]), linestyle='--',color=(0,0,0))
axs.plot(np.array([0, len(EVa)]), np.array([1,1]), linestyle='--',color=(0,0,0))
axs.set_yscale('log')
axs.set_xlim(0,len(EVa))
axs.set_ylim(np.min(EVa),np.max(EVa))
axs.set_xlabel('Index of Sigular Values',fontsize=20)
axs.set_ylabel('Sigular Values',fontsize=20)
axs.tick_params(axis='both', labelsize=20)
axs.grid(True)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs.text(0.05, 0.95, f"Number of Singular Value Larger than 1 is {idx_max}",
         transform=axs.transAxes, fontsize=20,
         verticalalignment='top')
axs.text(0.05, 0.9, f"Largest Singular Value is {np.max(EVa)}",
         transform=axs.transAxes, fontsize=20,
         verticalalignment='top')
fig.show()

PCA = edps_0s @ EVec
PCA = PCA.reshape((edps.shape[0], edps.shape[1], edps.shape[2]))
for idx in range(6):
    PCA_p = PCA[:,:,idx]
    fig = viz(EDPSam_Polar.geolocation, EDPSam_Polar.mesh,EDPSam_Polar.altitude,PCA_p.T,polar=True,
              title_txt=f"PCA #{idx}: ")
    plt.show()

for idx in range(50,150,20):
    PCA_p = PCA[:,:,idx]
    fig = viz(EDPSam_Polar.geolocation, EDPSam_Polar.mesh,EDPSam_Polar.altitude,PCA_p.T,polar=True,
              title_txt=f"PCA #{idx}: ")
    plt.show()

Sigma = np.std(edps_0s,axis=1)
Sigma = Sigma.reshape((edps.shape[0], edps.shape[1]))
fig = viz(EDPSam_Polar.geolocation, EDPSam_Polar.mesh,EDPSam_Polar.altitude,Sigma.T,polar=True,
              title_txt=f"PCA #{idx}: ")
plt.show()

Sigma_height = np.mean(Sigma,axis=1)
rng = np.random.default_rng()
RandChange = rng.normal(loc=0.0, scale=1.0, size=(edps.shape[0],edps.shape[1]*edps.shape[2]))
RandChange = Sigma_height[:,np.newaxis]*RandChange
RandChange = RandChange.reshape((edps.shape[0]*edps.shape[1],edps.shape[2]))

edps_pertb = edps_0s + RandChange
edps_0s = edps_pertb
CoV_inner = edps_0s.T @ edps_0s
EVa, EVec = np.linalg.eig(CoV_inner)
idx = np.argsort(EVa)[::-1]
EVa = EVa[idx]
EVec = EVec[:, idx]

EVa = EVa.reshape((edps_reshape.shape[1], 1))
EVa = np.abs(EVa)
EVa = np.sqrt(EVa)
idx = np.where(EVa>1)
idx = idx[0]
idx_max = max(idx)
#
fig = figure(figsize=(10, 10))
axs = fig.subplots(1, 1)
axs.plot(range(EVa.shape[0]), EVa, marker='o', linestyle='none')
axs.plot(np.array([idx_max, idx_max]), np.array([np.min(EVa),np.max(EVa)]), linestyle='--',color=(0,0,0))
axs.plot(np.array([0, len(EVa)]), np.array([1,1]), linestyle='--',color=(0,0,0))
axs.set_yscale('log')
axs.set_xlim(0,len(EVa))
axs.set_ylim(np.min(EVa),np.max(EVa))
axs.set_xlabel('Index of Sigular Values',fontsize=20)
axs.set_ylabel('Sigular Values',fontsize=20)
axs.tick_params(axis='both', labelsize=20)
axs.grid(True)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs.text(0.05, 0.95, f"Number of Singular Value Larger than 1 is {idx_max}",
         transform=axs.transAxes, fontsize=20,
         verticalalignment='top')
axs.text(0.05, 0.9, f"Largest Singular Value is {np.max(EVa)}",
         transform=axs.transAxes, fontsize=20,
         verticalalignment='top')
fig.show()

PCA = edps_0s @ EVec
PCA = PCA.reshape((edps.shape[0], edps.shape[1], edps.shape[2]))
for idx in range(6):
    PCA_p = PCA[:,:,idx]
    fig = viz(EDPSam_Polar.geolocation, EDPSam_Polar.mesh,EDPSam_Polar.altitude,PCA_p.T,polar=True,
              title_txt=f"PCA #{idx}: ")
    plt.show()

for idx in range(50,150,20):
    PCA_p = PCA[:,:,idx]
    fig = viz(EDPSam_Polar.geolocation, EDPSam_Polar.mesh,EDPSam_Polar.altitude,PCA_p.T,polar=True,
              title_txt=f"PCA #{idx}: ")
    plt.show()
