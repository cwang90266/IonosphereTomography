#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class of Tomography Data Assimilation Filter

Created on Mon May 11 13:47:34 2026

@author: cwang
"""

from filterpy.kalman import KalmanFilter
from edp_samples import EDPSamples as EDPS
import numpy as np

class Ionosphere_Tomography_Inverter(KalmanFilter):
    """
    The class Ionosphere_Tomography_Inverter is a subclass of the KalmanFilter 
    which is dedicated for ionosphere tomographic inversion from the total 
    electron content (TEC) measurements to electron density profiles. The key 
    difference between the standard Kalman filter object is that the dimension
    of the observation and the observation operator change in each iteration. 
    The state transition matrix is the identty matrix multiplied by a scalar
    coefficient exp(-delta). This is the Gauss Markov model for a stationary 
    process. 
    """
    def __init__(cls, EDPSam:EDPS,meanscale:int=None):
        edps = EDPSam.epds
        edps = edps.reshape(-1,edps.dim[2])
        if meanscale == 1:
            edps = edps/np.mean(edps,axis=1)

        initial_filter = KalmanFilter (dim_x=edps.dim[0], dim_z=1)
        initial_filter.x = np.zeros(edps.dim[0])
        initial_filter.P = np.cov(edps)
        initial_filter.attrs["meanscale"]=meanscale
        initial_filter.attrs["initial_edps"]=edps
        initial_filter.attrs["initial_edps_mean"]=np.mean(edps,axis=1)
        
        return initial_filter
    
    def assimilate(self, 
                   obs:np.array, 
                   obs_operator:np.array,
                   relaxation:float=1,
                   measurement_err:float=0) -> tuple[np.array,np.array,np.array]:
        assert obs.dim[0] == obs_operator.dim[0], "Row dimension of the observation operator must be equal to the dimension of the observation vector"
        assert obs_operator.dim[1] == self.x.dim[0], "Column dimension of the observation operator must be equal to the dimension of the state"
        
        new_filter = KalmanFilter (dim_x=self.x.dim[0], dim_z=obs.dim[0] )
        new_filter.P = self.P
        new_filter.x = self.x
        new_filter.attrs = self.attrs
        if self.meanscale == 1:
            obs_operator = obs_operator * self.attrs["initial_edps_mean"]
            
        new_filter.F = relaxation * np.eye(self.x.dim[0])
        new_filter.H = obs_operator
        
        intial_predict_obs = obs_operator @ self.attrs["initial_edps"]
        new_filter.R = np.cov(intial_predict_obs)+measurement_err*np.eye(obs.dim[0])
        new_filter.Q = (1- relaxation)*np.cov(self.attrs["initial_edps"])
        
        predict_obs = new_filter.predict()
        analysis_x = new_filter.update(obs-predict_obs)
        analysis_x = analysis_x+self.x
        if self.meanscale == 1:
            analysis_x = analysis_x * self.attrs["initial_edps_mean"]

        self = new_filter
        
        return analysis_x
        
        
            
        