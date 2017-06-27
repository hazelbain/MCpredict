# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:16:50 2017

@author: hazel.bain
"""

import pandas as pd
import MC_longterm as mcl
import richardson_mc_analysis as evtplt

t1 = ['1-jan-1998','1-jan-1999','1-jan-2000','1-jan-2001','1-jan-2002','1-jan-2003','1-jan-2004','1-jan-2005',\
      '1-jan-2006','1-jan-2007','1-jan-2008','1-jan-2009','1-jan-2010','1-jan-2011','1-jan-2012','1-jan-2013',\
      '1-jan-2014','1-jan-2015','1-jan-2016','1-jan-2017']
t2 = ['31-dec-1998','31-dec-1999','31-dec-2000','31-dec-2001','31-dec-2002','31-dec-2003','31-dec-2004','31-dec-2005',\
      '31-dec-2006','31-dec-2007','31-dec-2008','31-dec-2009','31-dec-2010','31-dec-2011','31-dec-2012','31-dec-2013',
      '31-dec-2014','31-dec-2015','31-dec-2016','31-may-2017']

events = pd.DataFrame()             #observational event characteristics for all MCs
events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
for i in range(len(t1)):
    
    events_tmp, events_frac_tmp = mcl.find_events(t1[i], t2[i], plotting=1, \
                                                  csv=1, livedb = 1)
    
    events = events.append(events_tmp)
    events_frac = events_frac.append(events_frac_tmp)

events.to_csv("events_1998_2017", sep='\t', encoding='utf-8') 
events_frac.to_csv("events_frac_1998_2017", sep='\t', encoding='utf-8')   

pickle.dump(events_frac,open("events_frac_1998_2017.p", "wb"))
pickle.dump(events,open("events_1998_2017.p", "w"))
    
evtplt.plot_obs_bz_tau(events, 'bzm_vs_tau_1998_2017.pdf')
evtplt.plot_predict_bz_tau_frac(events_frac, 'bztau_predict_1998_2017.pdf')


