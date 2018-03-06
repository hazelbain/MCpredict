# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:30:14 2017

@author: hazel.bain
"""

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
import read_kp as kp
import Richardson_ICME_list as icme

import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import calendar
import pickle as pickle
import platform

import MC_predict_pdfs as mcp
import MC_predict_plots as mcplt
from MCpredict import predict_geoeff, dst_geo_tag

import matplotlib.pyplot as plt

def train_and_validate(fname='', train_fname='', valid_fname='', \
                       train=1, trainfit=0, trainpdf=1, \
                       validfit=0, predict=1, report=1, \
                       ew=[2], nw=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], nbins=[50j,100j], \
                       fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],\
                       dst_thresh = -80, \
                       kp_thresh = 6, kp_thresh_old = 6, P1 = 0.2):
            
    ## TODO: add in fname for plot directories, missed and false

    if train == 1:
        
        #step 1: fit the training events
        if trainfit == 0:
            print("Loading the training data")
            ## TODO events_time_frac
            events_frac, events_time_frac = load_training_events(train_fname, ew[0], nw[0], \
                    dst_thresh=dst_thresh, kp_thresh = kp_thresh, kp_thresh_old = kp_thresh_old)
        else:
            print("Fitting the training data")
            events, events_frac, events_time_frac = fit_training_events(fname=fname, \
                    ew=ew[0], nw=nw[0], dst_thresh=dst_thresh, kp_thresh = kp_thresh)
            
            return events, events_frac, events_time_frac
            

        events_frac.tau_predicted.iloc[np.where((events_frac.frac == 0.0) & (events_frac.tau_predicted == np.inf))] = 0.0
        events_frac.drop_duplicates(('start','frac'), inplace = True)
        
        if trainpdf == 1:
            #### step 2: use the events_frac from above to generate the bayesian PDF
            print("Creating the PDFs")
            for e in ew:
                for n in nw:
                    pdf = mcp.create_pdfs(events_time_frac, ew=e, nw=n, nbins=nbins, fracs=fracs, \
                            fname=fname+"ew"+str(e)+"_nw"+str(n)+"_kp"+str(abs(kp_thresh)))
                
        #step 3: fit the validation events 
        if validfit == 0:
            print("Loading the validation data")
            events_frac_predict, events_time_frac_predict = load_validation_events(valid_fname, ew[0], nw[0], \
                    dst_thresh=dst_thresh, kp_thresh = kp_thresh, kp_thresh_old=kp_thresh_old)   
        else:
            print("Fitting the validation data")
            events_predict, events_frac_predict, events_time_frac_predict = fit_validation_events(fname=fname,\
                    ew=ew[0], nw=nw[0], dst_thresh = dst_thresh,  kp_thresh = kp_thresh)


        #return events, events_frac, events_predict, events_frac_predict
        return events_time_frac_predict

    else:
        #read in events
        print("Loading the validation data")
        events_frac_predict = pickle.load(open("valid/events_frac_"+valid_fname+"valid_ew"+str(ew[0])+"_nw"+str(nw[0])+"_kp"+str(abs(kp_thresh))+".p","rb"))
        events_time_frac_predict = pickle.load(open("valid/events_time_frac_"+valid_fname+"valid_ew"+str(ew[0])+"_nw"+str(nw[0])+"_kp"+str(abs(kp_thresh))+".p","rb"))


    events_frac_predict.drop_duplicates(('start','frac'), inplace = True)

    #step 4: prediction (once events are fittng we can skip the first step)
    if predict == 1:
        print("Predicting the geoeffectiveness")
        events_time_frac_predict2 = validate_events(events_time_frac_predict, fname=fname, \
                ew=ew, nw=nw, dst_thresh=dst_thresh, kp_thresh = kp_thresh)
               
    ##!!!!!!!!! check the boxplot to get the threshold P1 !!!!!!!!###
        
    ##write report
    if report == 1:
        print("Writing Report")
        for e in ew:
            for n in nw:
                events_frac_predict2 = pickle.load(open("valid/events_frac_"+fname+"predict_ew"+str(e)+"_nw"+str(n)+"_kp"+str(abs(kp_thresh))+".p","rb"))
                mcplt.write_report(events_frac_predict2, dd = "valid/plots/", fname=fname+"predict_ew"+str(e)+"_nw"+str(n)+"_kp"+str(abs(kp_thresh)), P1 = P1)
            
    return events_frac_predict2
    


def fit_training_events(fname = '', ew=2, nw=0.5, dst_thresh = -80, kp_thresh = 6, livedb = 0, csv = 0):

    #### step 1: gather events to use for the bayseian PDF, uses Chen_MC_prediction without predict keyword

    t1 = ['1-jan-1998','1-jan-1999','1-jan-2000','1-jan-2001','1-jan-2002','1-jan-2003', '1-jan-2004','1-jan-2005',\
          '1-jan-2006','1-jan-2007','1-jan-2008','1-jan-2009','1-jan-2010','1-jan-2011','1-jan-2012','1-jan-2013',\
          '1-jan-2014','1-jan-2015','1-jan-2016','1-jan-2017']

    
    t2 = ['31-dec-1998','31-dec-1999','31-dec-2000','31-dec-2001','31-dec-2002','31-dec-2003', '31-dec-2004','31-dec-2005',\
          '31-dec-2006','31-dec-2007','31-dec-2008','31-dec-2009','31-dec-2010','31-dec-2011','31-dec-2012','31-dec-2013', \
          '31-dec-2014','31-dec-2015','31-dec-2016','31-may-2017']
    
    #t1 = ['7-jan-2000']
    #t2 = ['13-jan-2000']
    
    events = pd.DataFrame()             #observational event characteristics for all MCs
    events_frac = pd.DataFrame()        #predicted events characteristics split into fraction of an event
    events_time_frac = pd.DataFrame()        #predicted events characteristics split into time increments
    for i in range(len(t1)):
        
        events_tmp, events_frac_tmp, events_time_frac_tmp = find_events(t1[i], t2[i], plotting=1, \
            dst_thresh = dst_thresh, kp_thresh = kp_thresh, csv=csv, livedb = livedb, fname = fname)
        
        #increment the evt_index by the number of events already held in events_frac, events_time_frac
        if len(events) > 0:
            events_frac_tmp.evt_index = events_frac_tmp.evt_index + (events_frac.evt_index.iloc[-1] + 1)
            events_time_frac_tmp.evt_index = events_time_frac_tmp.evt_index + (events_time_frac.evt_index.iloc[-1] + 1)        
        
        events = events.append(events_tmp)
        events_frac = events_frac.append(events_frac_tmp)
        events_time_frac = events_time_frac.append(events_time_frac_tmp)
     
    events = events.reset_index(drop=True) 
    events_frac = events_frac.reset_index(drop=True) 
    events_time_frac = events_time_frac.reset_index(drop=True) 
    
    #drop duplicate events 
    events_uniq = events.drop_duplicates()       
    events_frac_uniq = events_frac.drop_duplicates(['evt_index','start','frac_est','bzm_predicted','tau_predicted'])       
    events_time_frac_uniq = events_time_frac.drop_duplicates(['evt_index','start','frac_est','bzm_predicted','tau_predicted'])    
        
    #events.to_csv("train/events_"+fname+"train_kp"+str(abs(kp_thresh))+".csv", sep='\t', encoding='utf-8') 
    #events_frac.to_csv("train/events_frac_"+fname+"train_kp"+str(abs(kp_thresh))+".csv", sep='\t', encoding='utf-8')   
    
    pickle.dump(events_time_frac_uniq,open("train/events_time_frac_"+fname+"train_dst"+str(abs(dst_thresh))+"_kp"+str(abs(kp_thresh))+".p", "wb"))
    pickle.dump(events_frac_uniq,open("train/events_frac_"+fname+"train_dst"+str(abs(dst_thresh))+"_kp"+str(abs(kp_thresh))+".p", "wb"))
    pickle.dump(events_uniq,open("train/events_"+fname+"train_dst"+str(abs(dst_thresh))+"_kp"+str(abs(kp_thresh))+".p", "wb"))
        
    mcplt.plot_obs_bz_tau(events_uniq, dd = "train/plots/", fname = fname+"train_dst"+str(abs(dst_thresh))+"_kp"+str(abs(kp_thresh)))
    mcplt.plot_predict_bz_tau_frac(events_frac_uniq, dd = "train/plots/", fname = fname+"train_dst"+str(abs(dst_thresh))+"_kp"+str(abs(kp_thresh)))
    mcplt.plot_obs_vs_predict(events_frac_uniq, dd = "train/plots/", fname = fname+"train_dst"+str(abs(dst_thresh))+"_kp"+str(abs(kp_thresh)))
    #mcplt.plot_theta(events_frac_uniq, dd = "train/plots/", fname = fname+"train_kp"+str(abs(kp_thresh)))

    return events_uniq, events_frac_uniq, events_time_frac_uniq


def load_training_events(fname, ew, nw, dst_thresh=-80, kp_thresh = 6, kp_thresh_old = 6):

    #load the fitted events    
    events_frac = pickle.load(open("train/events_frac_"+fname+"train_kp"+str(abs(kp_thresh_old))+".p","rb")) 
    events_time_frac = pickle.load(open("train/events_time_frac_"+fname+"train_kp"+str(abs(kp_thresh_old))+".p","rb")) 


    if dst_thresh != kp_thresh_old:

        #read in the dst data
        dst_data = dst.read_dst_df()
        
        #read in kp data
        kp_data = kp.read_kp()
        
        #reset the events_frac index
        events_frac = events_frac.reset_index()
        events_time_frac = events_time_frac.reset_index()
        
        #get min dst and geoeffective flags and replace in value in dataframe
        geoeff, dstmin, dstdur = dst_geo_tag(events_frac, dst_data, dst_thresh = dst_thresh, \
                              dst_dur_thresh = 2, geoeff_only = 1)
    
        #replace geoeff column
        events_frac.geoeff = geoeff.geoeff
        events_frac.dst = dstmin
        events_frac.dstdur = dstdur
        
        #get min dst and geoeffective flags and replace in value in time dataframe
        geoeff_time, dstmin_time, dstdur_time = dst_geo_tag(events_time_frac, dst_data, dst_thresh = dst_thresh, \
                          dst_dur_thresh = 2, geoeff_only = 1)
    
        #replace geoeff column
        events_time_frac.geoeff = geoeff_time.geoeff
        events_time_frac.dst = dstmin_time
        events_time_frac.dstdur = dstdur_time
        
        #save the new dataframes
        #events_frac.to_csv("train/events_frac_"+fname+"train_dst"+str(abs(dst_thresh))+".csv", sep='\t', encoding='utf-8')   
        pickle.dump(events_frac,open("train/events_frac_"+fname+"train_dst"+str(abs(dst_thresh))+".p", "wb"))
        pickle.dump(events_time_frac,open("train/events_time_frac_"+fname+"train_dst"+str(abs(dst_thresh))+".p", "wb"))
            
    
    return events_frac, events_time_frac

def fit_validation_events(fname='', ew=2, nw=0.5, dst_thresh = -80):
    
    #### step 3: Use the generate PDFs to predict the geoeffectiveness of events
    #### in the remainder of the data set
    
    
    #restore prediction matrix

    
    pdf = pickle.load(open("PDFs/Pdict_"+fname+"ew"+str(ew)+"_nw"+str(nw)+"_dst"+str(abs(dst_thresh))+".p","rb"))
            
    t1 = ['1-jan-2004','1-jan-2005',\
          '1-jan-2006','1-jan-2007','1-jan-2008','1-jan-2009','1-jan-2010','1-jan-2011','1-jan-2012','1-jan-2013',\
          '1-jan-2014','1-jan-2015','1-jan-2016','1-jan-2017']
    t2 = ['31-dec-2004','31-dec-2005',\
          '31-dec-2006','31-dec-2007','31-dec-2008','31-dec-2009','31-dec-2010','31-dec-2011','31-dec-2012','31-dec-2013',
          '31-dec-2014','31-dec-2015','31-dec-2016','31-may-2017']
    
    #t1 = ['1-jan-2004']
    #t2 = ['31-jan-2004']

            
    events_predict = pd.DataFrame()             #observational event characteristics for all MCs
    events_frac_predict = pd.DataFrame()        #predicted events characteristics split into fraction of an event
    events_time_frac_predict = pd.DataFrame()        #predicted events characteristics split into time increments
    for i in range(len(t1)):
        
        events_tmp, events_frac_tmp, events_time_frac_tmp = find_events(t1[i], t2[i], pdf = pdf, plotting=1, \
                    ew=ew, nw=nw, dst_thresh = dst_thresh, csv=0, livedb = 0, predict = 0)
        
        #increment the evt_index by the number of events already held in events_frac, events_time_frac
        if len(events_predict) > 0:
            events_frac_tmp.evt_index = events_frac_tmp.evt_index + (events_frac_predict.evt_index.iloc[-1] + 1)
            events_time_frac_tmp.evt_index = events_time_frac_tmp.evt_index + (events_time_frac_predict.evt_index.iloc[-1] + 1)        
        
        events_predict = events_predict.append(events_tmp)
        events_frac_predict = events_frac_predict.append(events_frac_tmp)
        events_time_frac_predict = events_time_frac_predict.append(events_time_frac_tmp)
        
    events_predict = events_predict.reset_index(drop=True) 
    events_frac_predict = events_frac_predict.reset_index(drop=True) 
    events_time_frac_predict = events_time_frac_predict.reset_index(drop=True)  
    
    #drop duplicate events 
    events_predict_uniq = events_predict.drop_duplicates()       
    events_frac_predict_uniq = events_frac_predict.drop_duplicates(['evt_index','start','frac_est','bzm_predicted','tau_predicted'])       
    events_time_frac_predict_uniq = events_time_frac_predict.drop_duplicates(['evt_index','start','frac_est','bzm_predicted','tau_predicted'])    

    
    #events_predict.to_csv("valid/events_"+fname+"valid_ew"+str(ew)+"_nw"+str(nw)+"_dst"+str(abs(dst_thresh))+".csv", sep='\t', encoding='utf-8') 
    #events_frac_predict.to_csv("valid/events_frac"+fname+"valid_ew"+str(ew)+"_nw"+str(nw)+"_dst"+str(abs(dst_thresh))+".csv", sep='\t', encoding='utf-8')   
    
    pickle.dump(events_time_frac_predict_uniq,open("valid/events_time_frac_"+fname+"valid_ew"+str(ew)+"_nw"+str(nw)+"_dst"+str(abs(dst_thresh))+".p", "wb"))
    pickle.dump(events_frac_predict_uniq,open("valid/events_frac_"+fname+"valid_ew"+str(ew)+"_nw"+str(nw)+"_dst"+str(abs(dst_thresh))+".p", "wb"))
    pickle.dump(events_predict_uniq,open("valid/events_"+fname+"valid_ew"+str(ew)+"_nw"+str(nw)+"_dst"+str(abs(dst_thresh))+".p", "wb"))
                
    #plots
    #mcplt.plot_obs_bz_tau(events_predict, dd = "valid/plots/", fname = fname+"_valid_ew"+str(ew)+"_nw"+str(nw))
    #mcplt.plot_predict_bz_tau_frac(events_frac_predict, dd = "valid/plots/", fname = fname+"_valid_ew"+str(ew)+"_nw"+str(nw))
    #mcplt.plot_obs_vs_predict(events_frac_predict, dd = "valid/plots/", fname= fname+'_valid_ew'+str(ew)+'_nw'+str(nw))
    #mcplt.plot_bzm_vs_tau_skill(events_frac_predict, dd = "valid/plots/", P1 = 0.1, fname=fname+'_valid_ew'+str(ew)+'_nw'+str(nw))
    #mcplt.plot_theta(events_frac_predict, dd = "valid/plots/", fname = fname+'_valid_ew'+str(ew)+'_nw'+str(nw))
        
    return events_predict_uniq, events_frac_predict_uniq, events_time_frac_predict_uniq


def load_validation_events(fname, ew, nw, dst_thresh=-80, dst_thresh_old = -80):

    #load the fitted events    
    events_frac_predict = pickle.load(open("valid/events_frac_"+fname+"valid_ew"+str(ew)+"_nw"+str(nw)+"_dst"+str(abs(dst_thresh_old))+".p","rb")) 
    events_time_frac_predict = pickle.load(open("valid/events_time_frac_"+fname+"valid_ew"+str(ew)+"_nw"+str(nw)+"_dst"+str(abs(dst_thresh_old))+".p","rb")) 

    if dst_thresh != dst_thresh_old:

        #read in the dst data
        dst_data = dst.read_dst_df()
        
        #reset the events_frac index
        events_frac_predict = events_frac_predict.reset_index()
        
        #get min dst and geoeffective flags and replace in value in dataframe
        geoeff, dstmin, dstdur = dst_geo_tag(events_frac_predict, dst_data, dst_thresh = dst_thresh, \
                              dst_dur_thresh = 2, geoeff_only = 1)
    
        #replace geoeff column
        events_frac_predict.geoeff = geoeff.geoeff
        events_frac_predict.dst = dstmin
        events_frac_predict.dstdur = dstdur
        
        #save the new dataframes
        #events_frac_predict.to_csv("valid/events_frac_"+fname+"valid_ew"+str(ew)+"_nw"+str(nw)+"_dst"+str(abs(dst_thresh))+".csv", sep='\t', encoding='utf-8')   
        pickle.dump(events_frac_predict,open("valid/events_frac_"+fname+"valid_ew"+str(ew)+"_nw"+str(nw)+"_dst"+str(abs(dst_thresh))+".p", "wb"))
            
    
    return events_frac_predict, events_time_frac_predict



def validate_events(events_time_frac_predict, fname='', ew=[2], nw=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dst_thresh = -80):

    #### step 3a rerun predict witout reading in the data again
    
    for e in ew:
        for n in nw:
            
            #restore prediction matrix
            pdf = pickle.load(open("PDFs/Pdict_"+fname+"ew"+str(e)+"_nw"+str(n)+"_dst"+str(abs(dst_thresh))+".p","rb")) 
            
            #first strip the predict from event_frac_predict
            cols_to_keep = ['evt_index', 'data_index','start', 'bzm', 'tau', 'istart', 'iend',\
                            'end', 'dst', 'dstdur', 'geoeff', 'bzm_predicted', 'frac', 'frac_est', 'i_bzmax', \
                            'tau_predicted','theta_z_max','dtheta_z']
            events_time_frac = events_time_frac_predict.filter(cols_to_keep,axis=1)
            
            #repredict the geoeffectiveness without refitting Bz 
            events_time_frac_predict2 = predict_geoeff(events_time_frac, pdf)

            #save
            #events_frac_predict2.to_csv("valid/events_frac_predict_"+fname+"valid_ew"+str(e)+"_nw"+str(n)+"_dst"+str(abs(dst_thresh))+".csv", sep='\t', encoding='utf-8')   
            pickle.dump(events_time_frac_predict2,open("valid/events_time_frac_"+fname+"predict_ew"+str(e)+"_nw"+str(n)+"_dst"+str(abs(dst_thresh))+".p", "wb"))
            
            #plots
            #mcplt.plot_obs_bz_tau(events_predict, dd = "valid/plots/", fname = fname+"_valid_ew"+str(ew)+"_nw"+str(nw))
            #mcplt.plot_predict_bz_tau_frac(events_frac_predict, dd = "valid/plots/", fname = fname+"_valid_ew"+str(ew)+"_nw"+str(nw))
            #mcplt.plot_obs_vs_predict(events_frac_predict, dd = "valid/plots/", fname=fname+'_valid_ew'+str(ew)+'_nw'+str(nw))
            #mcplt.plot_bzm_vs_tau_skill(events_frac_predict, dd = "valid/plots/", P1 = 0.1, fname=fname+'_valid_ew'+str(ew)+'_nw'+str(nw))
            #mcplt.plot_bzmp_vs_taup_skill(events_frac_predict, dd = "valid/plots/", P1 = 0.1, fname=fname+'_valid_ew'+str(ew)+'_nw'+str(nw))
            
            #boxplot
            mcplt.plot_boxplot(events_time_frac_predict2, dd = 'valid/plots/', fname = fname+'predict_ew'+str(e)+'_nw'+str(n)+"_dst"+str(abs(dst_thresh)))

    
    return events_time_frac_predict2





def find_events(start_date, end_date, plotting = 0, csv = 1, livedb = 0, 
                predict = 0, ew = 2, nw = 1, dst_thresh = -80, kp_thresh = 6,\
                fname = '', pdf = np.zeros((50,50,50,50))):

    if platform.system() == 'Darwin':
        proj_dir = '/Users/hazelbain/Dropbox/MCpredict/MCpredict/'
    else:
        proj_dir = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/'
    
    
    #format times
    start_date = datetime.strptime(start_date, "%d-%b-%Y")
    end_date= datetime.strptime(end_date, "%d-%b-%Y")
    
    if (end_date.year - start_date.year) > 0:
        print("*** Dates need to be in the same calander year ***")
        return None
    
    else:
            
        #read in the dst data
        dst_data = dst.read_dst_df()
        
        #read in the kp data
        kp_data = kp.read_kp()

        #read in Richardson and Cane ICME list 
        icme_list = icme.read_richardson_icme_list()

        
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
        events_time_frac = pd.DataFrame()        #predicted events characteristics split into time increments through an event
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

                    #start and end of ICME
                    line1 = list(icme_list.query('plasma_start >="'+stf+'" and plasma_start <= "'+ datetime.strftime(et + timedelta(days = 1), "%Y-%m-%d") +'"')\
                                [['plasma_start','plasma_end']].values.flatten())

                    #start and end of MC
                    line2 = list(icme_list.query('mc_start >="'+stf+'" and mc_start <= "'+ datetime.strftime(et + timedelta(days = 1), "%Y-%m-%d") +'"')\
                                [['mc_start','mc_end']].values.flatten())
                    
                    data, events_tmp, events_frac_tmp, events_time_frac_tmp = MC.Chen_MC_Prediction(stf, etf, \
                        dst_data[st - timedelta(1):et + timedelta(1)], \
                        kp_data[st - timedelta(1):et + timedelta(1)], \
                        line = line1, line2=line2,\
                        pdf = pdf, csv = csv, livedb = livedb, predict = predict,\
                        smooth_num = 100, dst_thresh = dst_thresh, kp_thresh = kp_thresh, plotting = plotting,\
                        plt_outfile = 'mcpredict_'+ datetime.strftime(date_list[i][0], "%Y-%m-%d_%H%M") + '.pdf' ,\
                        plt_outpath = proj_dir + 'longterm_'+fname[0:-1]+'/')
                    
                    #increment the evt_index by the number of events already held in events_frac, events_time_frac    
                    if len(events) > 0:
                        events_frac_tmp.evt_index = events_frac_tmp.evt_index + (events_frac.evt_index.iloc[-1] + 1)
                        events_time_frac_tmp.evt_index = events_time_frac_tmp.evt_index + (events_time_frac.evt_index.iloc[-1] + 1)
     
                    events = events.append(events_tmp)
                    events_frac = events_frac.append(events_frac_tmp)
                    events_time_frac = events_time_frac.append(events_time_frac_tmp)

                except:
                    print("*** Error running Chen MC Prediction ***")
                    errpredict.append(i)

        events = events.reset_index() 
        events_frac = events_frac.reset_index() 
        events_time_frac = events_time_frac.reset_index() 
    
        #drop duplicate events 
        events_uniq = events.drop_duplicates()       
        events_frac_uniq = events_frac.drop_duplicates(['evt_index','start','frac_est','bzm_predicted','tau_predicted'])       
        events_time_frac_uniq = events_time_frac.drop_duplicates(['evt_index','start','frac_est','bzm_predicted','tau_predicted'])      

        
        print("--------Error predict------------")
        print(errpredict)
    
        #plot_obs_bz_tau(events_uniq, 'bzm_vs_tau_smooth100.pdf')
        #plot_predict_bz_tau_frac(events_frac, outname = 'bztau_predict.pdf')
        
        return events_uniq, events_frac_uniq, events_time_frac_uniq




            