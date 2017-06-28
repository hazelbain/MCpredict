# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:16:50 2017

@author: hazel.bain
"""
%load_ext autoreload
%autoreload 2


import pickle as pickle
import pandas as pd
import MC_longterm as mcl
import MC_predict_pdfs as mcp
import richardson_mc_analysis as evtplt


#### step 1: gather events to use for the bayseian PDF, uses Chen_MC_prediction without predict keyword

t1 = ['1-jan-1998','1-jan-1999','1-jan-2000','1-jan-2001','1-jan-2002','1-jan-2003']
t2 = ['31-dec-1998','31-dec-1999','31-dec-2000','31-dec-2001','31-dec-2002','31-dec-2003']

#t1 = ['1-jan-2005']
#t2 = ['14-jan-2005']

events = pd.DataFrame()             #observational event characteristics for all MCs
events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
for i in range(len(t1)):
    
    events_tmp, events_frac_tmp = mcl.find_events(t1[i], t2[i], plotting=0, \
                                                  csv=0, livedb = 1)
    
    events = events.append(events_tmp)
    events_frac = events_frac.append(events_frac_tmp)

events.to_csv("events_1998_2004.csv", sep='\t', encoding='utf-8') 
events_frac.to_csv("events_frac_1998_2004.csv", sep='\t', encoding='utf-8')   

pickle.dump(events_frac,open("events_frac_1998_2004.p", "wb"))
pickle.dump(events,open("events_1998_2004.p", "wb"))
    
evtplt.plot_obs_bz_tau(events, 'bzm_vs_tau_1998_2004.pdf')
evtplt.plot_predict_bz_tau_frac(events_frac, 'bztau_predict_1998_2004.pdf')


#### step 2: use the events_frac from above to generate the bayesian PDF

pdf = mcp.create_pdfs(events_frac, fname='1998_2004' )

#restore prediction matrix
#pdf = pickle.load(open("Pdict_ew2_nw1.p","rb"))

#### step 3: Use the generate PDFs to predict the geoeffectiveness of events
#### in the remainder of the data set

t1 = ['1-jan-2004','1-jan-2005',\
      '1-jan-2006','1-jan-2007','1-jan-2008','1-jan-2009','1-jan-2010','1-jan-2011','1-jan-2012','1-jan-2013',\
      '1-jan-2014','1-jan-2015','1-jan-2016','1-jan-2017']
t2 = ['31-dec-2004','31-dec-2005',\
      '31-dec-2006','31-dec-2007','31-dec-2008','31-dec-2009','31-dec-2010','31-dec-2011','31-dec-2012','31-dec-2013',
      '31-dec-2014','31-dec-2015','31-dec-2016','31-may-2017']

events_predict = pd.DataFrame()             #observational event characteristics for all MCs
events_frac_predict = pd.DataFrame()        #predicted events characteristics split into fraction of an event
for i in range(len(t1)):
    
    events_tmp, events_frac_tmp = mcl.find_events(t1[i], t2[i], plotting=0, \
                                                  csv=0, livedb = 1)
    
    events_predict = events.append(events_tmp)
    events_frac_predict = events_frac.append(events_frac_tmp)

events_predict.to_csv("events_2004_2017.csv", sep='\t', encoding='utf-8') 
events_frac_predict.to_csv("events_frac_2004_2017.csv", sep='\t', encoding='utf-8')   

pickle.dump(events_frac_predict,open("events_frac_2004_2017.p", "wb"))
pickle.dump(events_predict,open("events_2004_2017.p", "wb"))
    
evtplt.plot_obs_bz_tau(events_predict, 'bzm_vs_tau_2004_2017.pdf')
evtplt.plot_predict_bz_tau_frac(events_frac_predict, 'bztau_predict_2004_2017.pdf')







#==============================================================================
# t1 = ['1-jan-1998','1-jan-1999','1-jan-2000','1-jan-2001','1-jan-2002','1-jan-2003','1-jan-2004','1-jan-2005',\
#       '1-jan-2006','1-jan-2007','1-jan-2008','1-jan-2009','1-jan-2010','1-jan-2011','1-jan-2012','1-jan-2013',\
#       '1-jan-2014','1-jan-2015','1-jan-2016','1-jan-2017']
# t2 = ['31-dec-1998','31-dec-1999','31-dec-2000','31-dec-2001','31-dec-2002','31-dec-2003','31-dec-2004','31-dec-2005',\
#       '31-dec-2006','31-dec-2007','31-dec-2008','31-dec-2009','31-dec-2010','31-dec-2011','31-dec-2012','31-dec-2013',
#       '31-dec-2014','31-dec-2015','31-dec-2016','31-may-2017']
# 
#==============================================================================
