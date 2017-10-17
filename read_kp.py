# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:06:07 2017

@author: hazel.bain
"""
import numpy as np
import pandas as pd

from urllib.request import urlopen
from datetime import datetime


def read_kp(csv = 0):

    data = []
    for year in range(1994,2018): 
        
        urlfile = 'ftp://ftp.ngdc.noaa.gov/STP/GEOMAGNETIC_DATA/INDICES/KP_AP/' + str(year)
        
        #open and read data files
        txt = urlopen(urlfile).read()
        txt2 = txt.decode("utf-8").split('\n')[0:-1]
        

        for row in txt2:
            for c in np.arange(12,27,2):

                #kp major scaling
                if row[c:c+1] == ' ':
                    kp_major = 0
                else:
                    kp_major = int(row[c:c+1])
                
                #kp minor sclaing i.e. m/z/p   
                if row[c+1:c+2] == ' ':
                    kp_minor = 0
                else:
                    kp_minor = int(row[c+1:c+2])
                
                #join to make full kp value
                kp_tmp = kp_major + (kp_minor/10)
                
                #corespoinding observation time
                if int(row[0:2]) > 50:
                    time_tmp = datetime(int(row[0:2])+1900, int(row[2:4]), int(row[4:6]), int((c-12)/2*3))
                else:
                    time_tmp = datetime(int(row[0:2])+2000, int(row[2:4]), int(row[4:6]), int((c-12)/2*3))
                
                data.append((time_tmp,kp_tmp))


    #convert to pandas datafram     
    data_df = pd.DataFrame.from_records(data, columns = ['date','kp'])
        
    #save as csv file
    if csv == 1:
        data_df.to_csv('.spyproject/kp.csv', index_label = 'index')
    
    return data_df





