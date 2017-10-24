# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:00:13 2017

"""
#format time strings

from datetime import datetime, timedelta
import MCpredict as MC
import read_dst as dst
import read_kp as kp
import pickle as pickle

pdf = pickle.load(open("PDFs/Pdict_peturb_ew2_nw0.6_dst80.p","rb"))

#st = datetime(1998, 11, 7)
#et = datetime(1998, 11, 9)

#st = datetime(1998, 6, 23)
#et = datetime(1998, 6, 27)

#st = datetime(2004, 1, 21)
#`et = datetime(2004, 1, 25)

#st = datetime(2004, 2, 11)
#et = datetime(2004, 2, 13)

#st = datetime(1998, 11, 8)
#et = datetime(1998, 11, 12)

#st = datetime(2006, 5, 24)
#et = datetime(2006, 5, 25)

#st = datetime(2003, 10, 28)
#et = datetime(2003, 10, 31)

st = datetime(1998, 4, 27)
et = datetime(1998, 5, 6)

stf = datetime.strftime(st, "%Y-%m-%d")
etf = datetime.strftime(et, "%Y-%m-%d")

#read in the dst data
dst_data = dst.read_dst_df()

#read in the kp data
kp_data = kp.read_kp()
                                                 
data, events_tmp, events_frac_tmp, events_time_frac_tmp = MC.Chen_MC_Prediction(stf, etf, \
        dst_data[st - timedelta(1):et + timedelta(1)], \
        kp_data[st - timedelta(1):et + timedelta(1)], \
        pdf = pdf, csv = 0, livedb = 0  , predict = 1,\
        smooth_num = 100, plotting = 0,\
        plt_outfile = 'test12.pdf' ,\
        plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/')
                    
