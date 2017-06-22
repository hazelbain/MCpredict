# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:16:50 2017

@author: hazel.bain
"""

t1 = ['1-jan-1998','1-jan-1999','1-jan-2000','1-jan-2001','1-jan-2002','1-jan-2003','1-jan-2004','1-jan-2005']
t2 = ['31-dec-1998','31-dec-1999','31-dec-2000','31-dec-2001','31-dec-2002','31-dec-2003','31-dec-2004','31-dec-2005']

events = pd.DataFrame()             #observational event characteristics for all MCs
events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
for i in range(len(t1)):
    
    events_tmp, events_frac_tmp = mcl.find_events(t1[i], t2[i], plotting=1, \
                                                  csv=1, livedb = 1)
    
    events = events.append(events_tmp)
    events_frac = events_frac.append(events_frac_tmp)