# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:00:13 2017

"""
#format time strings

from datetime import datetime, timedelta
import MCpredict as MC
import read_dst as dst
import pickle as pickle

pdf = pickle.load(open("Pdict_ew2_nw0.5_th2_1998_2004.p","rb"))

#st = datetime(1998, 11, 7)
#et = datetime(1998, 11, 9)

#st = datetime(1998, 6, 23)
#et = datetime(1998, 6, 27)

#st = datetime(2004, 1, 21)
#`et = datetime(2004, 1, 25)

st = datetime(2004, 2, 11)
et = datetime(2004, 2, 13)

stf = datetime.strftime(st, "%Y-%m-%d")
etf = datetime.strftime(et, "%Y-%m-%d")

#read in the dst data
dst_data = dst.read_dst_df()
                                                 
data, events_tmp, events_frac_tmp = MC.Chen_MC_Prediction(stf, etf, \
        dst_data[st - timedelta(1):et + timedelta(1)], pdf = pdf, \
        csv = 0, livedb = 1  , predict = 0,\
        smooth_num = 100, plotting = 1,\
        plt_outfile = 'test6.pdf' ,\
        plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/')
                    
