# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:12:25 2017

@author: hazel.bain
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:05:15 2017

@author: hazel.bain
"""
import MCpredict as MC
import read_dst as dst

import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import calendar
import pickle as pickle

import MC_predict_pdfs as mcp
import MC_predict_plots as mcplt
from MCpredict import predict_geoeff 


def train_and_validate(fname=fname, revalidate=1, ew=[2], nw=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    
    fname = 'th3'
    
    ew=[2]
    nw=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    #train events
    events, events_frac, pdf = train_events(fname=fname, ew=ew, nw=nw)
    
    if revalidate == 0:
        #validate events
        events_predict, events_frac_predict = validate_events(pdf, fname=fname,\
                    ew=ew, nw=nw)
    else:
        #reevaluate
        events_frac_predict = pickle.load(open("events_frac_predict_th3_2004_2017.p","rb"))
        events_frac_predict2 = revalidate_events(events_frac_predict, fname=fname, \
                    ew=ew, nw=nw, P1 = 0.2)
    
    
    
    ##!!!!!!!!! check the boxplot to get the threshold P1 !!!!!!!!###
    
    ## TODO: Add ew and nw to boxplot name
    
    ##write report
    write_report(events_frac, outname = 'mc_predict_test_results', fname = '', P1 = 0.2)


def train_events(fname = '', ew=[2], nw=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):

    #### step 1: gather events to use for the bayseian PDF, uses Chen_MC_prediction without predict keyword

    t1 = ['1-jan-1998','1-jan-1999','1-jan-2000','1-jan-2001','1-jan-2002','1-jan-2003']
    t2 = ['31-dec-1998','31-dec-1999','31-dec-2000','31-dec-2001','31-dec-2002','31-dec-2003']
    
    #t1 = ['1-jan-2005']
    #t2 = ['7-jan-2005']
    
    events = pd.DataFrame()             #observational event characteristics for all MCs
    events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
    for i in range(len(t1)):
        
        events_tmp, events_frac_tmp = find_events(t1[i], t2[i], plotting=1, \
                                                      csv=0, livedb = 1)
        
        events = events.append(events_tmp)
        events_frac = events_frac.append(events_frac_tmp)
        
    events = events.drop_duplicates()       
    events_frac = events_frac.drop_duplicates()      
        
    events.to_csv("events_"+fname+"_1998_2004.csv", sep='\t', encoding='utf-8') 
    events_frac.to_csv("events_frac_"+fname+"_1998_2004.csv", sep='\t', encoding='utf-8')   
    
    pickle.dump(events_frac,open("events_frac_"+fname+"_1998_2004.p", "wb"))
    pickle.dump(events,open("events_"+fname+"_1998_2004.p", "wb"))
        
    mcplt.plot_obs_bz_tau(events, 'bzm_vs_tau_'+fname+'_1998_2004.pdf')
    mcplt.plot_predict_bz_tau_frac(events_frac, 'bztau_predict_'+fname+'_1998_2004.pdf')
    mcplt.plot_obs_vs_predict(events_frac, fname=fname+'_1998_2004')
    
    #### step 2: use the events_frac from above to generate the bayesian PDF
    
    for e in ew:
        for n in nw:
            pdf = mcp.create_pdfs(events_frac, ew=e, nw=n, fname=fname+'_1998_2004' )
    

    return events, events_frac, pdf



def validate_events(pdf, fname='', ew=[2], nw=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    
    #### step 3: Use the generate PDFs to predict the geoeffectiveness of events
    #### in the remainder of the data set
    
    #restore prediction matrix
    for e in ew:
        for n in nw:
    
            pdf = pickle.load(open("Pdict_ew"+str(ew)+"_nw"+str(nw)+"_"+fname+".p","rb"))
            
            t1 = ['1-jan-2004','1-jan-2005',\
                  '1-jan-2006','1-jan-2007','1-jan-2008','1-jan-2009','1-jan-2010','1-jan-2011','1-jan-2012','1-jan-2013',\
                  '1-jan-2014','1-jan-2015','1-jan-2016','1-jan-2017']
            t2 = ['31-dec-2004','31-dec-2005',\
                  '31-dec-2006','31-dec-2007','31-dec-2008','31-dec-2009','31-dec-2010','31-dec-2011','31-dec-2012','31-dec-2013',
                  '31-dec-2014','31-dec-2015','31-dec-2016','31-may-2017']

            
            events_predict = pd.DataFrame()             #observational event characteristics for all MCs
            events_frac_predict = pd.DataFrame()        #predicted events characteristics split into fraction of an event
            for i in range(len(t1)):
                
                events_tmp, events_frac_tmp = find_events(t1[i], t2[i], pdf = pdf, plotting=1, \
                                                    csv=0, livedb = 1, predict = 1)
                
                events_predict = events_predict.append(events_tmp)
                events_frac_predict = events_frac_predict.append(events_frac_tmp)
            
            events_predict = events_predict.drop_duplicates()       
            events_frac_predict = events_frac_predict.drop_duplicates()  
            
            ##TODO: Add ew and nw to save names
            
            events_predict.to_csv("events_"+fname+"_2004_2017.csv", sep='\t', encoding='utf-8') 
            events_frac_predict.to_csv("events_frac_"+fname+"_2004_2017.csv", sep='\t', encoding='utf-8')   
            
            pickle.dump(events_frac_predict,open("events_frac_"+fname+"_2004_2017.p", "wb"))
            pickle.dump(events_predict,open("events_"+fname+"_2004_2017.p", "wb"))
                
            #mcplt.plot_obs_bz_tau(events_predict, 'bzm_vs_tau_th3_2004_2017.pdf')
            #mcplt.plot_predict_bz_tau_frac(events_frac_predict, 'bztau_predict_th3_2004_2017.pdf')
            #mcplt.plot_obs_vs_predict(events_frac_predict, fname='th3_2004_2017')
            #mcplt.plot_bzm_vs_tau_skill(events_frac_predict, P1 = 0.1, fname = 'th3_2004_2017')
            #mcplt.plot_bzmp_vs_taup_skill(events_frac_predict, P1 = 0.1, fname = 'th3_2004_2017')
            
            mcplt.plot_boxplot(events_frac_predict, fname = fname+'_2004_2017')
    
    return events_predict, events_frac_predict


def revalidate_events(events_frac_predict, fname='', ew=[2], nw=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], P1 = 0.2):

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

            #save
            events_frac_predict2.to_csv("events_frac_"+fname+"_2004_2017_"+str(ew)+"_nw"+str(nw)+".csv", sep='\t', encoding='utf-8')   
            pickle.dump(events_frac_predict2,open("events_frac_"+fname+"_2004_2017_"+str(ew)+"_nw"+str(nw)+".p", "wb"))
            
            #boxplot
            mcplt.plot_boxplot(events_frac_predict2, fname = fname+'_2004_2017_'+str(ew)+'_nw'+str(nw))

    
    return events_frac_predict2





def find_events(start_date, end_date, plotting = 0, csv = 1, livedb = 0, 
                predict = 0, ew = 2, nw = 1, pdf = np.zeros((50,50,50,50))):

    #format times
    start_date = datetime.strptime(start_date, "%d-%b-%Y")
    end_date= datetime.strptime(end_date, "%d-%b-%Y")
    
    if (end_date.year - start_date.year) > 0:
        print("*** Dates need to be in the same calander year ***")
        return None
    
    else:
            
        #read in the dst data
        dst_data = dst.read_dst_df()
        
        #get list of week start and end dates - with overlap of one day
        date_list = []
        cal = calendar.Calendar()
        for y in (np.arange(end_date.month - start_date.month + 1)+start_date.month):
            for x in cal.monthdatescalendar(start_date.year, y):
                date_list.append([x[0], x[0]+timedelta(days = 8)])
        date_list = np.asarray(date_list)
        
        #print(start_date.month, end_date.month)   
        #print(np.arange(end_date.month - start_date.month + 1)+start_date.month)
        #print(date_list)
      
    
        #get the ace_mag_1m and ace_swepam_1m data for these events
        events = pd.DataFrame()             #observational event characteristics for all MCs
        events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
        errpredict = []
        for i in range(0,len(date_list)):
                    
            #get mc times +/- 24 hours
            st = datetime.combine(date_list[i,0], datetime.min.time())
            et = datetime.combine(date_list[i,1], datetime.min.time())
            
            if st >= start_date - timedelta(days = 7) and st <= end_date:
                
                #format time strings
                stf = datetime.strftime(st, "%Y-%m-%d")
                etf = datetime.strftime(et, "%Y-%m-%d")
                
                try:
                                                 
                    data, events_tmp, events_frac_tmp = MC.Chen_MC_Prediction(stf, etf, \
                        dst_data[st - timedelta(1):et + timedelta(1)], pdf = pdf, \
                        csv = csv, livedb = livedb, predict = predict,\
                        smooth_num = 100, plotting = plotting,\
                        plt_outfile = 'mcpredict_'+ datetime.strftime(date_list[i][0], "%Y-%m-%d_%H%M") + '.pdf' ,\
                        plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/longterm_th3/')
                    
                    events = events.append(events_tmp)
                    events_frac = events_frac.append(events_frac_tmp)
    
                    
                except:
                    print("*** Error running Chen MC Prediction ***")
                    errpredict.append(i)
    
        events = events.reset_index() 
    
        #drop duplicate events 
        events_uniq = events.drop_duplicates()       
        events_frac_uniq = events_frac.drop_duplicates()       
                
        print("--------Error predict------------")
        print(errpredict)
    
        #plot_obs_bz_tau(events_uniq, 'bzm_vs_tau_smooth100.pdf')
        #plot_predict_bz_tau_frac(events_frac, outname = 'bztau_predict.pdf')
        
        return events_uniq, events_frac_uniq




            