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
import MC_predict_plots as mcplt
from MCpredict import predict_geoeff 


#### step 1: gather events to use for the bayseian PDF, uses Chen_MC_prediction without predict keyword

t1 = ['1-jan-1998','1-jan-1999','1-jan-2000','1-jan-2001','1-jan-2002','1-jan-2003']
t2 = ['31-dec-1998','31-dec-1999','31-dec-2000','31-dec-2001','31-dec-2002','31-dec-2003']

#t1 = ['1-jan-2005']
#t2 = ['7-jan-2005']

events = pd.DataFrame()             #observational event characteristics for all MCs
events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
for i in range(len(t1)):
    
    events_tmp, events_frac_tmp = mcl.find_events(t1[i], t2[i], plotting=1, \
                                                  csv=0, livedb = 1)
    
    events = events.append(events_tmp)
    events_frac = events_frac.append(events_frac_tmp)
    
events = events_predict.drop_duplicates()       
events_frac = events_frac.drop_duplicates()      
    
events.to_csv("events_th_1998_2004.csv", sep='\t', encoding='utf-8') 
events_frac.to_csv("events_frac_th_1998_2004.csv", sep='\t', encoding='utf-8')   

pickle.dump(events_frac,open("events_frac_th_1998_2004.p", "wb"))
pickle.dump(events,open("events_th_1998_2004.p", "wb"))
    
mcplt.plot_obs_bz_tau(events, 'bzm_vs_tau_th_1998_2004.pdf')
mcplt.plot_predict_bz_tau_frac(events_frac, 'bztau_predict_th_1998_2004.pdf')
mcplt.plot_obs_vs_predict(events_frac, fname='th_1998_2017')

#### step 2: use the events_frac from above to generate the bayesian PDF

pdf = mcp.create_pdfs(events_frac, fname='th_1998_2004' )

#restore prediction matrix
#pdf = pickle.load(open("Pdict_ew2_nw1.p","rb"))

#### step 3: Use the generate PDFs to predict the geoeffectiveness of events
#### in the remainder of the data set


#pdf = pickle.load(open("Pdict_ew2_nw0.51998_2004.p","rb"))

t1 = ['1-jan-2004','1-jan-2005',\
      '1-jan-2006','1-jan-2007','1-jan-2008','1-jan-2009','1-jan-2010','1-jan-2011','1-jan-2012','1-jan-2013',\
      '1-jan-2014','1-jan-2015','1-jan-2016','1-jan-2017']
t2 = ['31-dec-2004','31-dec-2005',\
      '31-dec-2006','31-dec-2007','31-dec-2008','31-dec-2009','31-dec-2010','31-dec-2011','31-dec-2012','31-dec-2013',
      '31-dec-2014','31-dec-2015','31-dec-2016','31-may-2017']


#t1 = ['1-jan-2005']
#t2 = ['14-jan-2005']

events_predict = pd.DataFrame()             #observational event characteristics for all MCs
events_frac_predict = pd.DataFrame()        #predicted events characteristics split into fraction of an event
for i in range(len(t1)):
    
    events_tmp, events_frac_tmp = mcl.find_events(t1[i], t2[i], pdf = pdf, plotting=1, \
                                        csv=0, livedb = 1, predict = 1)
    
    events_predict = events_predict.append(events_tmp)
    events_frac_predict = events_frac_predict.append(events_frac_tmp)

events_predict = events_predict.drop_duplicates()       
events_frac_predict = events_frac_predict.drop_duplicates()  

events_predict.to_csv("events_th_2004_2017.csv", sep='\t', encoding='utf-8') 
events_frac_predict.to_csv("events_frac_th_2004_2017.csv", sep='\t', encoding='utf-8')   

pickle.dump(events_frac_predict,open("events_frac_th_2004_2017.p", "wb"))
pickle.dump(events_predict,open("events_th_2004_2017.p", "wb"))
    
mcplt.plot_obs_bz_tau(events_predict, 'bzm_vs_tau_th_2004_2017.pdf')
mcplt.plot_predict_bz_tau_frac(events_frac_predict, 'bztau_predict_th_2004_2017.pdf')







#### step 3a rerun predict witout reading in the data again

#first strip the predict from event_frac_predict
cols_to_keep = ['evt_index', 'data_index','start', 'bzm', 'tau', 'istart', 'iend',\
                'end', 'dst', 'dstdur', 'geoeff', 'bzm_predicted', 'frac', 'i_bzmax', \
                'tau_predicted']
events_frac = events_frac_predict.filter(cols_to_keep,axis=1)
events_frac_predict2 = predict_geoeff(events_frac, pdf)

#### Run some numbers
w = np.where(events_frac_predict2.frac == 1.0)[0]

#boxplot
events_frac_predict2.boxplot(column = 'P1_scaled', by = 'geoeff')

#contingency table
thresh = 0.2
pd.crosstab(events_frac_predict2.geoeff.iloc[w] == 1.0, events_frac_predict2.P1_scaled.iloc[w]> thresh)

#skill score
a = len(np.where((events_frac_predict2.geoeff.iloc[w] == 1.0) & (events_frac_predict2.P1_scaled.iloc[w]> thresh))[0])
b = len(np.where((events_frac_predict2.geoeff.iloc[w] == 0.0) & (events_frac_predict2.P1_scaled.iloc[w]> thresh))[0])
c = len(np.where((events_frac_predict2.geoeff.iloc[w] == 1.0) & (events_frac_predict2.P1_scaled.iloc[w]< thresh))[0])

CSI = a / (a+b+c)

#incorrect events
missed, false = sort_incorrect(events_frac_predict2)

##plots
mcplt.plot_bzm_vs_tau_skill(events_frac_predict2, fname = '2004_2017')



#obs vs predicted
evts0 = events_frac0[['bzm','bzm_predicted','tau','tau_predicted']]\
                    .iloc[np.where((events_frac0.frac == 1.0) & (events_frac0.geoeff == 1))[0]]
                    
evts = events_frac[['bzm','bzm_predicted','tau','tau_predicted']]\
                    .iloc[np.where((events_frac.frac == 1.0) & (events_frac.geoeff == 1))[0]]                    

t_mean_diff0 = np.mean(np.abs(evts0.tau - evts0.tau_predicted)) 
t_stdev_diff0 = np.std(np.abs(evts0.tau - evts0.tau_predicted)) 

t_mean_diff = np.mean(np.abs(evts.tau - evts.tau_predicted)) 
t_stdev_diff = np.std(np.abs(evts.tau - evts.tau_predicted)) 


#



def sort_incorrect(events_frac):
    
    import shutil
    import calendar
    
    #missed = events_frac_predict2.query('geoeff == 1.0 and frac == 1.0 and P1_scaled < 0.2').sort_values(by='dst',ascending=1)[['start','dst','P1_scaled']]
    #false = events_frac_predict2.query('geoeff == 0.0 and frac == 1.0 and P1_scaled > 0.2').sort_values(by='dst',ascending=1)[['start','dst','P1_scaled']]

    missed = events_frac_predict2.query('geoeff == 1.0 and frac == 1.0 and P1_scaled < 0.2').sort_values(by='start')[['start','dst','P1_scaled']]
    false = events_frac_predict2.query('geoeff == 0.0 and frac == 1.0 and P1_scaled > 0.2').sort_values(by='start')[['start','dst','P1_scaled']]
    
    dd_longterm = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/longterm/'
    dd_missed = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/missed/'
    dd_false = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/false/'
    
    for i in range(len(missed)):
                
        #construct filename
        year = missed.start.iloc[i].year
        mnth = missed.start.iloc[i].month                
        cal = calendar.Calendar()
        week_begin = [j[0] for j in cal.monthdatescalendar(year, mnth)]
        fdate = week_begin[np.max(np.where(missed.start.iloc[i].date() >= np.asarray(week_begin)))]
        missed_str = dd_longterm + 'mcpredict_'+ datetime.datetime.strftime(fdate, '%Y-%m-%d') + '_0000.pdf'
        new_loc = dd_missed + 'mcpredict_'+ datetime.datetime.strftime(fdate, '%Y-%m-%d') + '_0000.pdf'
        
        shutil.copyfile( missed_str, new_loc) 
        
    for i in range(len(false)):
                
        #construct filename
        year = missed.start.iloc[i].year
        mnth = missed.start.iloc[i].month                
        cal = calendar.Calendar()
        week_begin = [j[0] for j in cal.monthdatescalendar(year, mnth)]
        fdate = week_begin[np.max(np.where(missed.start.iloc[i].date() >= np.asarray(week_begin)))]
        false_str = dd_longterm + 'mcpredict_'+ datetime.datetime.strftime(fdate, '%Y-%m-%d') + '_0000.pdf'
        new_loc = dd_false + 'mcpredict_'+ datetime.datetime.strftime(fdate, '%Y-%m-%d') + '_0000.pdf'
        
        shutil.copyfile( missed_str, new_loc) 

    return missed, false
    

