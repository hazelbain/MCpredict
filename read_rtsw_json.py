#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:31:41 2018

@author: hazelbain
"""

import pandas as pd


def read_rtmag_3day_json():    


    mag_data = pd.read_json('http://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json')
    names = list(mag_data.iloc[0])
    mag_data.drop([0],inplace=True)
    mag_data.columns = names
    
    return mag_data


def read_rtsw_3day_json():    

    sw_data = pd.read_json('http://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json')
    names = list(sw_data.iloc[0])
    sw_data.drop([0],inplace=True)
    sw_data.columns = names
    sw_data.rename(columns={'density': 'n', 'speed': 'v', 'temperature':'t'}, inplace=True)
    
    return sw_data