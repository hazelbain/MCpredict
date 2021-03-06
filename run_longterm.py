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


def train_events():

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
        
    events = events.drop_duplicates()       
    events_frac = events_frac.drop_duplicates()      
        
    events.to_csv("events_th3_1998_2004.csv", sep='\t', encoding='utf-8') 
    events_frac.to_csv("events_frac_th3_1998_2004.csv", sep='\t', encoding='utf-8')   
    
    pickle.dump(events_frac,open("events_frac_th3_1998_2004.p", "wb"))
    pickle.dump(events,open("events_th3_1998_2004.p", "wb"))
        
    mcplt.plot_obs_bz_tau(events, 'bzm_vs_tau_th3_1998_2004.pdf')
    mcplt.plot_predict_bz_tau_frac(events_frac, 'bztau_predict_th3_1998_2004.pdf')
    mcplt.plot_obs_vs_predict(events_frac, fname='th3_1998_2004')
    
    #### step 2: use the events_frac from above to generate the bayesian PDF
    
    for ew in [2]:
        for nw in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            pdf = mcp.create_pdfs(events_frac, ew=ew, nw=nw, fname='th3_1998_2004' )
    

    return events, events_frac



def validate_events(pdf = pdf):
    
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
    
    events_predict.to_csv("events_th3_2004_2017.csv", sep='\t', encoding='utf-8') 
    events_frac_predict.to_csv("events_frac_th3_2004_2017.csv", sep='\t', encoding='utf-8')   
    
    pickle.dump(events_frac_predict,open("events_frac_th3_2004_2017.p", "wb"))
    pickle.dump(events_predict,open("events_th3_2004_2017.p", "wb"))
        
    #mcplt.plot_obs_bz_tau(events_predict, 'bzm_vs_tau_th3_2004_2017.pdf')
    #mcplt.plot_predict_bz_tau_frac(events_frac_predict, 'bztau_predict_th3_2004_2017.pdf')
    #mcplt.plot_obs_vs_predict(events_frac_predict, fname='th3_2004_2017')
    #mcplt.plot_bzm_vs_tau_skill(events_frac_predict, P1 = 0.1, fname = 'th3_2004_2017')
    #mcplt.plot_bzmp_vs_taup_skill(events_frac_predict, P1 = 0.1, fname = 'th3_2004_2017')
    
    return events_predict, events_frac_predict


def revalidate_events(events_frac_predict, ew=ew, nw=nw):

    #### step 3a rerun predict witout reading in the data again
    
    for e in ew:
        for n in nw:
            
            #restore prediction matrix
            pdf = pickle.load(open("Pdict_ew"+str(ew)+"_nw"+str(nw)+"_"+fname+".p","rb"))
            
            #first strip the predict from event_frac_predict
            cols_to_keep = ['evt_index', 'data_index','start', 'bzm', 'tau', 'istart', 'iend',\
                            'end', 'dst', 'dstdur', 'geoeff', 'bzm_predicted', 'frac', 'i_bzmax', \
                            'tau_predicted']
            events_frac = events_frac_predict.filter(cols_to_keep,axis=1)
            
            #repredict the geoeffectiveness without refitting Bz 
            events_frac_predict2 = predict_geoeff(events_frac, pdf)
    
    return events_frac_predict2



#boxplot
w = np.where(events_frac_predict.frac == 1.0)[0]
ax = events_frac_predict.boxplot(column = 'P1_scaled', by = 'geoeff')
fig = ax.get_figure()
fig.savefig('P1_boxplot_th3_2004_2017.pdf', format = 'pdf')
plt.close('all')


#contingency table
#thresh = 0.1
#pd.crosstab(events_frac_predict.geoeff.iloc[w] == 1.0, events_frac_predict.P1_scaled.iloc[w]> thresh)







