#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:17:09 2018

@author: hazelbain
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:24:48 2018

@author: hazelbain
"""

import read_dst as dst
import read_kp as kp
import Richardson_ICME_list as icme

from read_database import get_data
from MCpredict import clean_data

import pickle as pickle
import platform
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import calendar


def main():
    
    if platform.system() == 'Darwin':
        proj_dir = '/Users/hazelbain/Dropbox/MCpredict/MCpredict/'
    else:
        proj_dir = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/'
    
    
    ## read in fit all events files
    events_time_frac = pickle.load(open("train/events_time_frac_fitall3_train_dst80_kp6_clean2.p","rb"))
    events = pickle.load(open("train/events_fitall3_train_dst80_kp6_clean2.p","rb"))
    
    events_time_frac = events_time_frac.drop_duplicates(["evt_index","start","frac_start"], keep='last')
    
    #read in the dst data
    dst_data = dst.read_dst_df()
        
    #read in kp data
    kp_data = kp.read_kp()
    
    #read in Richardson and Cane ICME list 
    icme_list = icme.read_richardson_icme_list()
    
    #read in the pdfs
    Pdict2 = pickle.load(open("PDFs/Pdict_30interp_100_75_2.p","rb"))
    #Pdict2 = pickle.load(open("PDFs/Pdictn_events_time_frac_fitall3_train_dst80_kp6_clean2_ss0.p","rb"))
    
    
    
    time_before = 24

    #loop through clean MC events i.e. geoeff == 1
    geo1_ind = (events.query('geoeff == 1').index).astype('int')
    
    for g in geo1_ind[-14:-1]:
           
        #format times
        st = datetime.strptime(str(events.start.iloc[g]), "%Y-%m-%d %H:%M:%S") - timedelta(seconds = time_before*60*60)
        et= datetime.strptime(str(events.end.iloc[g]), "%Y-%m-%d %H:%M:%S") + timedelta(seconds = 3*60*60)
                
        #format time strings
        stf = datetime.strftime(st, "%Y-%m-%d")
        etf = datetime.strftime(et, "%Y-%m-%d")
                
        #print("Start date: " + stf )
        #print("End date  : " + etf + "\n")   

        #get the ACE data
        data, data_mag, data_sw = get_data_for_plots(stf, etf)
                    
        #subset of events and events_frac
        events_time_frac_ss = events_time_frac.query('evt_index == '+str(g))
        
        evtfname = datetime.strftime(events_time_frac_ss.start.iloc[0], "%Y%m%d_%H%M") 
        print("\n event: %s \n" % evtfname)
        if not os.path.isdir("case_studies/goodmc/"+evtfname):
            os.makedirs("case_studies/goodmc/"+evtfname)
        
        #print(events_time_frac_ss[["evt_index","start","frac_start","bzm_predicted"]])
                    
        #start and end of event times
        evt_times = events.iloc[g][['start','end']].values

        #start and end of ICME
        line1 = list(icme_list.query('plasma_start >="'+stf+'" and plasma_start <= "'+ datetime.strftime(et + timedelta(days = 1), "%Y-%m-%d") +'"')\
                     [['plasma_start','plasma_end']].values.flatten())
                    
        #start and end of MC
        line2 = list(icme_list.query('mc_start >="'+stf+'" and mc_start <= "'+ datetime.strftime(et + timedelta(days = 1), "%Y-%m-%d") +'"')\
                     [['mc_start','mc_end']].values.flatten())
              

        for i in range(len(events_time_frac_ss)):
        #for i in [0,20]:    
        #i = 10
            
        
            stplot = st 
            etplot = st +timedelta(seconds = (time_before*60*60 + i*60*60) )
            
            #print(etplot)
            
            data_ss = data.query('date >= "' + datetime.strftime(stplot, "%d-%b-%Y %H:%M:%S") + '" and date <= "'+ \
                                 datetime.strftime(etplot, "%d-%b-%Y %H:%M:%S")+'"')
            
            #print("data")
            #print(data_ss.date.iloc[0], data_ss.date.iloc[-1])
            
            dst_data_ss = dst_data[stplot:etplot]
            kp_data_ss = kp_data[stplot:etplot]
        
            #print(i)
        
            #make the plot
            mcmock_plot(data_ss, events_time_frac_ss, \
                       dst_data_ss[st - timedelta(1):et + timedelta(1)], \
                       kp_data_ss[st - timedelta(1):et + timedelta(1)], \
                       Pdict2, \
                       et, line=line1, line2=line2, bars = evt_times, \
                       plt_outfile = evtfname+"_"+str(i).zfill(3)+".pdf",\
                       plt_outpath = "case_studies/goodmc/"+evtfname+"/")
        
        print("done evt")

                    
    #return data, data_mag, data_sw
    return events_time_frac_ss

def mcmock_plot(data, events_time_frac, dst_data, kp_data, Pdict, et, line=[], line2=[], bars = [],\
            plot_fit = 1, dst_thresh = -80, kp_thresh = 6, \
            plt_outpath = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/richardson_mcpredict_plots_2/',\
            plt_outfile = 'mcpredict.pdf'):
    
    """
    Plot the ACE_MAG_1m and ACE_SWEPAM_1M data

    Parameters
    ----------
    tstart : string, required 
        Start time for the database query.
    tend: string, required 
        End time for the database query.
    server: string, optional
        default server is swds-st
    database: string, optional
        default database is RA
    view: string, optional
        default view is ace_mag_1m
    csv: int, optional
        output csv file keyword. default is 0
    outpath: string, optional
        csv file path

    Returns
    -------
    None
    
    """
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    from matplotlib.font_manager import FontProperties
    from matplotlib.dates import DayLocator
    from matplotlib.dates import HourLocator, MinuteLocator, AutoDateLocator
    from matplotlib.dates import DateFormatter
    from matplotlib.ticker import MultipleLocator
    from matplotlib.gridspec import GridSpec
    from matplotlib.font_manager import FontProperties
            
    #start and end times for plot to make sure all plots are consistent
    #st = datetime.strptime(data['date'][0]), "%Y-%m-%d")
    #et = datetime.strptime(data['date'][-1], "%Y-%m-%d")


    st = data['date'].iloc[0]
    #et = data['date'].iloc[-1]
    
    #read in the dst data
    #dst_data = dst.read_dst(str(st), str(et))

    #current parameters
    first_evt_index = int(events_time_frac.index[0])
    #print("first ind %i " % first_evt_index)
    curr_ind = events_time_frac[events_time_frac.frac_start <= data.date.iloc[-1]].index.astype('int').max() - first_evt_index
    #print("curr ind %i " % curr_ind)
    pred_dur_curr = events_time_frac['tau_predicted'].iloc[curr_ind] * 60. 
    pred_bz_curr = events_time_frac['bzm_predicted'].iloc[curr_ind]
    end_time_curr = events_time_frac['frac_start'].iloc[curr_ind] + timedelta(seconds = pred_dur_curr*60.)

    ##probs
    
    indt = []
    indb = []
    
    #print(curr_ind)
    if curr_ind == 0:
        est_frac = events_time_frac['frac_est'].iloc[0]
        pred_dur = events_time_frac['tau_predicted'].iloc[0]
        pred_bz = events_time_frac['bzm_predicted'].iloc[0]
        indt.append(np.min(np.where(Pdict["indices"][3,:] > pred_dur))) 
        indb.append(np.max(np.where(Pdict["indices"][2,:] < pred_bz)))
        #if est_frac < 0.2=:
        P0 = Pdict["prob_e"][indb, indt,0]
        
        #print("indexes")
        #print(indb)
        #print(indt)
    else:
        est_frac = events_time_frac['frac_est'].iloc[0:curr_ind]
        pred_dur = events_time_frac['tau_predicted'].iloc[0:curr_ind]
        pred_bz = events_time_frac['bzm_predicted'].iloc[0:curr_ind]
        
        #print(len(pred_dur))
        for i in range(len(pred_dur)):
            #print(pred_dur[i])
            #print(pred_bz[i])
            #print(i)
            indt.append(np.min(np.where(Pdict["indices"][3,:] > pred_dur[i])))
            indb.append(np.max(np.where(Pdict["indices"][2,:] < pred_bz[i])))
        
            #print(indb)
            #print(indt)
            #indt = [ np.min(np.where(Pdict["indices"][3,:] > pd)) for pd in pred_dur.values] 
            #indb = [ np.max(np.where(Pdict["indices"][2,:] < pbz)) for pbz in pred_bz.values]
            #print("indexes")
            #print(indt)
            #print(indb)
        #if est_frac < 0.2=:
        P0 = Pdict["prob_e"][indb, indt,0]
        

    #print("est_frac")
    #print(est_frac)
    #print("pred_dur")
    #print(pred_dur)
    #print("pred_dur_ind")
    #print(indt)
    #print("P0")
    #print(P0)
    #print("\n")


    #plot the ace data
    f = plt.figure(figsize=(11,8.5))
    #f, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(7, figsize=(12,10))
    
    fontP = FontProperties()				#legend
    fontP.set_size('small')
    
    ##Annotate sections
    fs = 10
    vv = 0.86
    
    plt.gcf().text(0.8, vv, 'Current Params', fontsize=12)

    plt.gcf().text(0.8, vv-0.04, 'Cur Time: %s' % datetime.strftime(data.date.iloc[-1], "%d-%b %H:%M:%S"), fontsize=fs)
    plt.gcf().text(0.8, vv-0.08, 'Bx: %.2f (nT)' % data.bx.iloc[-1], fontsize=fs)
    plt.gcf().text(0.8, vv-0.10, 'By: %.2f (nT)' % data.by.iloc[-1], fontsize=fs)
    plt.gcf().text(0.8, vv-0.12, 'Bz: %.2f (nT)' % data.bz.iloc[-1], fontsize=fs)
    plt.gcf().text(0.8, vv-0.16, 'n:  %.2f ($\mathrm{cm^-3}$)' % data.sw_n.iloc[-1], fontsize=fs)
    plt.gcf().text(0.8, vv-0.18, 'v:  %.2f ($\mathrm{km s^-1}$)' % data.sw_v.iloc[-1], fontsize=fs)
        
    plt.gcf().text(0.8, vv-0.24, 'Predicted Params', fontsize=fs)
    plt.gcf().text(0.8, vv-0.26, 'Max |Bz|: %.2f (nT)' % pred_bz_curr, fontsize=fs)
    plt.gcf().text(0.8, vv-0.28, 'Duration: %.2f (hrs)' % (pred_dur_curr/60.), fontsize=fs) 
    plt.gcf().text(0.8, vv-0.30, 'End Time: %s' % datetime.strftime(end_time_curr, "%d-%b %H:%M:%S"), fontsize=fs)
        
    vv2 = 0.24
    plt.gcf().text(0.8, vv2, 'Geoeffective Storm Probability', fontsize=fs)
    plt.gcf().text(0.8, vv2-0.02, 'P0:     %.4f (percent)' % P0[-1] , fontsize=fs)
    plt.gcf().text(0.8, vv2-0.04, 'P1:     XXX' , fontsize=fs)
    plt.gcf().text(0.8, vv2-0.06, 'P2:     XXX' , fontsize=fs)
    plt.gcf().text(0.8, vv2-0.08, 'P3:     XXX' , fontsize=fs)
    
    
    gs1 = GridSpec(7, 1)
    gs1.update(top=0.9, bottom=0.33, left = 0.1, right = 0.75)
    ax0 = plt.subplot(gs1[0,:])
    ax1 = plt.subplot(gs1[1,:])
    ax2 = plt.subplot(gs1[2,:])
    ax3 = plt.subplot(gs1[3,:])
    ax4 = plt.subplot(gs1[4,:])
    ax5 = plt.subplot(gs1[5,:])
    ax6 = plt.subplot(gs1[6,:])

    gs2 = GridSpec(1, 3)
    gs2.update(top=0.25, bottom=0.15, left = 0.1, right = 0.75)    
    ax00 = plt.subplot(gs2[:,0])
    ax11 = plt.subplot(gs2[:,1])
    ax22 = plt.subplot(gs2[:,2])
    
    gs3 = GridSpec(1, 1)
    gs3.update(top=0.6, bottom=0.25, left = 0.8, right = 0.95)    
    ax000 = plt.subplot(gs3[:,0])

    plt.subplots_adjust(hspace = .1)       # no vertical space between subplots
    fontP = FontProperties()                #legend
    fontP.set_size('medium')
    
    dateFmt = DateFormatter('%d-%b')
    dateFmt_sub = DateFormatter('%H:%M')
    hoursLoc = HourLocator()
    daysLoc = DayLocator()
    minsLoc = MinuteLocator()
    
    #plt.tight_layout()
    
    minorLocator = MultipleLocator(1)
    
    
    color = {0.0:'green', 1.0:'red', 2.0:'grey',3.0:'orange'}
    #fitcolor = {0.2:'purple', 0.4:'blue', events_frac.frac.iloc[3]:'green',0.8:'orange', 1.0:'red'}

    #----Bx
    ax0.plot(data['date'], data['bx'], c='black', label='Bx (nT)')
    ax0.hlines(0.0, data['date'].iloc[0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax0.set_xticklabels(' ')
    ax0.set_ylabel('Bx (nT)')
    ax0.xaxis.set_major_locator(daysLoc)
    ax0.xaxis.set_minor_locator(hoursLoc)
    ax0.set_xlim([st, et])
    ax0.yaxis.set_minor_locator(minorLocator)
    for l in line:
        ax0.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax0.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    #for b in range(len(bars)):
    #    ax0.axvspan(bars[b,0], bars[b,1], facecolor=color[events['geoeff'].iloc[b]], alpha=0.15)        
    #leg = ax0.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    #leg.get_frame().set_alpha(0.5)

  
    #----By
    ax1.plot(data['date'], data['by'], c='black', label='By (nT)')
    ax1.hlines(0.0, data['date'].iloc[0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax1.set_xticklabels(' ')
    ax1.set_ylabel('By (nT)')
    ax1.xaxis.set_major_locator(daysLoc)
    ax1.xaxis.set_minor_locator(hoursLoc)
    ax1.set_xlim([st, et])
    ax1.yaxis.set_minor_locator(minorLocator)
    for l in line:
        ax1.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax1.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    #for b in range(len(bars)):
    #    ax1.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15)        
    #leg = ax1.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    #leg.get_frame().set_alpha(0.5)

        
    #----Bz
    ax2.plot(data['date'], data['bz'], label='Bz (nT)', c='r')
    ax2.hlines(0.0, data['date'].iloc[0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax2.set_xticklabels(' ')
    ax2.set_ylabel('Bz (nT)')
    ax2.xaxis.set_major_locator(daysLoc)
    ax2.xaxis.set_minor_locator(hoursLoc)
    ax2.set_xlim([st, et])
    ax2.yaxis.set_minor_locator(minorLocator)
    for l in line:
        ax2.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax2.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    #for b in range(len(bars)):
    #    ax2.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax2.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
#==============================================================================
#     #plot the position of max bz
#     for i in np.arange(5, len(events_frac), 6):
#         if (events_frac['geoeff'].iloc[i] == 1.0):
#             wmax_bz = np.where( data['bz'].iloc[events_frac['istart_bz'].iloc[i] : events_frac['iend_bz'].iloc[i]] == events_frac['bzm'].iloc[i])[0]
# 
#             ax1.axvline(x=data['date'].iloc[events_frac['istart_bz'].iloc[i] + wmax_bz].values[0], \
#                      linewidth=1, linestyle='--', color='grey')
#==============================================================================

#==============================================================================
#     #max bz line
#     for b in range(len(bars)):
#         if events_frac['geoeff'].iloc[b*6] == 1.0:
#             ax1.hlines(events_frac['bzm'].iloc[b*6], bars[b,0], bars[b,1], linestyle='-',color='grey')
#==============================================================================

    #plot the fitted profile  
    fit_times = [ events_time_frac['start'].iloc[curr_ind] + timedelta(seconds = j*60) for j in np.arange(pred_dur_curr)]
    fit_profile = events_time_frac['bzm_predicted'].iloc[curr_ind] * np.sin(np.pi*np.arange(0,1,1./(pred_dur_curr)) )          
                
    ax2.plot(fit_times, fit_profile, c='black', ls = '--')    

    #----density
    ax3.plot(data['date'], data['sw_n'], c='orange', label='n ($\mathrm{cm^-3}$)')
    ax3.set_xticklabels(' ')
    ax3.set_ylabel('n ($\mathrm{cm^-3}$)')
    ax3.xaxis.set_major_locator(daysLoc)
    ax3.xaxis.set_minor_locator(hoursLoc)
    ax3.set_xlim([st, et])
    for l in line:
        ax3.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax3.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    #for b in range(len(bars)):
    #    ax3.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    #leg = ax3.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    #leg.get_frame().set_alpha(0.5)
    
    #----velocity
    #maxv = max(  data['sw_v'].loc[np.where(np.isnan(data['sw_v']) == False )] ) + 50
    #minv =  min(  data['sw_v'].loc[np.where(np.isnan(data['sw_v']) == False )] ) - 50
    
    maxv = np.nanmax(data.sw_v.values)
    minv = np.nanmin(data.sw_v.values)
    ax4.plot(data['date'], data['sw_v'], c='yellow', label='v ($\mathrm{km s^-1}$)')
    ax4.set_ylim(top = maxv, bottom = minv)
    ax4.set_xticklabels(' ')
    ax4.set_ylabel('v ($\mathrm{km s^-1}$)')
    ax4.xaxis.set_major_locator(daysLoc)
    ax4.xaxis.set_minor_locator(hoursLoc)
    ax4.set_xlim([st, et])
    for l in line:
        ax4.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax4.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    #for b in range(len(bars)):
    #    ax4.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15)       
    #leg = ax4.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    #leg.get_frame().set_alpha(0.5)
            
    #----dst
    ax5.plot(dst_data[st:et].index, dst_data[st:et]['dst'], c='black', label='Dst')
    ax5.hlines(dst_thresh, data['date'].iloc[0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax5.set_xticklabels(' ')
    ax5.set_ylabel('Dst (nT)')
    ax5.xaxis.set_major_locator(daysLoc)
    ax5.xaxis.set_minor_locator(hoursLoc)
    ax5.set_xlim([st, et])
    for l in line:
        ax5.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax5.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    #for b in range(len(bars)):
    #    ax5.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['dstgeoeff'].iloc[b*6]], alpha=0.15) 
    #ax7.set_xlabel("Start Time "+ str(st)+" (UTC)")
    #leg = ax5.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    #leg.get_frame().set_alpha(0.5)
    
    
    #--- plot kp
    x0 = mdates.date2num(kp_data.index[0])
    x1 = mdates.date2num(kp_data.index[1])
    y=kp_data.kp.iloc[0]
    w=x1-x0 
    if y < 4.0:
        barcolor = 'green'
    elif y >= 4.0 and y < 5.0:
        barcolor = 'orange'
    elif y >= 5.0:
        barcolor = 'red'
    ax6.bar(x0, y, width = w, color = barcolor, edgecolor='black', align = 'edge', label='Kp')

    for i in range(len(kp_data)-1):
        x0 = mdates.date2num(kp_data.index[i])
        x1 = mdates.date2num(kp_data.index[i+1])
        y=kp_data.kp.iloc[i]
        w=x1-x0 
        if y < 4.0:
            barcolor = 'green'
        elif y >= 4.0 and y < 5.0:
            barcolor = 'orange'
        elif y >= 5.0:
            barcolor = 'red'
        ax6.bar(x0, y, width = w, color = barcolor, edgecolor='black', align = 'edge')

    ax6.hlines(kp_thresh, data['date'].iloc[0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax6.set_xticklabels(' ')
    ax6.set_ylabel('Kp')
    ax6.xaxis.set_major_formatter(dateFmt)
    ax6.xaxis.set_major_locator(daysLoc)
    ax6.xaxis.set_minor_locator(hoursLoc)
    ax6.set_xlim([st, et])
    ax6.set_ylim(0,10)
    for l in line:
        ax6.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax6.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    #for b in range(len(bars)):
    #    ax6.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    ax6.set_xlabel("Start Time "+ str(st)+" (UTC)")
    #leg = ax6.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    #leg.get_frame().set_alpha(0.5)        


    #predicted parameters plots
    
    #only plot the data from the event
    curr_evt = events_time_frac['evt_index'].iloc[curr_ind] 
    #first_evt_index = int(events_time_frac.query('evt_index == '+str(curr_evt) ).index[0])
    
    if curr_ind == 0:
        frac_time = [events_time_frac.frac_start.iloc[0]]
        frac_bz_pred = [events_time_frac.bzm_predicted.iloc[0]]
        frac_tau_pred = [events_time_frac.tau_predicted.iloc[0]]
    else:
        frac_time = events_time_frac.frac_start.iloc[0:curr_ind]
        frac_bz_pred = events_time_frac.bzm_predicted.iloc[0:curr_ind]
        frac_tau_pred = events_time_frac.tau_predicted.iloc[0:curr_ind]
    
    myFmt = mdates.DateFormatter('%H:%M')
    
    #----Bz pred
    ax00.plot(frac_time, frac_bz_pred, c='red', label='n ($\mathrm{cm^-3}$)')
    #ax00.set_ylabel('Bz pred ($\mathrm{cm^-3}$)', fontsize=8)
    ax00.set_title('Bz pred (nT)', fontsize=8)
    ax00.tick_params(labelsize=8)
    ax00.xaxis.set_major_formatter(myFmt)
    #plt.gcf().autofmt_xdate()
    plt.setp(ax00.xaxis.get_majorticklabels(), rotation=70 )
    
    #ax00.axis('tight')
    #ax00.xaxis.set_major_locator(daysLoc)
    #ax00.xaxis.set_minor_locator(hoursLoc)
    #ax00.set_xlim([st, et])
    
    #----Bz pred
    ax11.plot(frac_time, frac_tau_pred, c='black', label='n ($\mathrm{cm^-3}$)')
    #ax11.set_ylabel('Dur pred ($\mathrm{cm^-3}$)', fontsize=8)
    ax11.set_title('Dur pred ($\mathrm{cm^-3}$)', fontsize=8)
    ax11.tick_params(labelsize=8)
    ax11.xaxis.set_major_formatter(myFmt)
    plt.setp(ax11.xaxis.get_majorticklabels(), rotation=70 )
    #ax11.gca().autofmt_xdate()
    #ax11.axis('tight')
    #ax11.xaxis.set_major_locator(hoursLoc)
    #ax11.xaxis.set_minor_locator(minsLoc)
    #ax11.xaxis.set_minor_formatter(dateFmt_sub)
    #locator = AutoDateLocator()
    #ax11.set_locator(locator)
    #ax00.set_xlim([st, et])

    
    #----Bz pred
    #ax22.axis('off')
    ax22.plot(frac_time, P0, c='orange', label='n ($\mathrm{cm^-3}$)')
    #ax22.set_ylabel('Bz pred ($\mathrm{cm^-3}$)')
    ax22.set_title('Prob Geoeff (P0 %)', fontsize=8)
    ax22.tick_params(labelsize=8)
    ax22.xaxis.set_major_formatter(myFmt)
    plt.setp(ax22.xaxis.get_majorticklabels(), rotation=70 )
    #ax22.xaxis.set_major_locator(daysLoc)
    #ax22.xaxis.set_minor_locator(hoursLoc)
    #ax22.xaxis.set_major_formatter(dateFmt_sub)
    #ax00.set_xlim([st, et])
    #ax22.gcf().autofmt_xdate()

    #gs1.tight_layout()
    gs2.tight_layout(f, rect=(0.075,0.1,0.76,0.27), pad=0.0)
    

    ##prob_e
    ranges = Pdict["ranges"]
    c1 = ax000.imshow(np.rot90(Pdict["prob_e"][:,:,0]), extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
    ax000.plot(frac_bz_pred, frac_tau_pred, markersize=4, c='r', label = 'bzm, tau, g = 1')
    #ax1.plot(gbzm, gtau, 'k.', markersize=4, c='r', label = 'bzm, tau, g = 1')
    ax000.set_xlim([ranges[0], ranges[1]])
    ax000.set_ylim([0, ranges[3]])
    ax000.set_xlabel('Bzm')
    ax000.set_ylabel('Tau')
    ax000.set_title('Prob Geoeffective')
    f.colorbar(c1, ax = ax000, fraction=0.025)



    #plt.show()
    plt.savefig(plt_outpath + plt_outfile, format='pdf')    #, bbox_inches='tight', pad_inches=0)

    plt.close()          
          
    
    return None



def get_data_for_plots(sdate, edate, csv = 0, livedb = 0):
    

    #read in mag and solar wind data

    #read in ace_mag_1m data
    mag_data = get_data(sdate, edate, view = 'ace_mag_1m', \
                            csv = csv, livedb = livedb)
    
    sw_data = get_data(sdate, edate, view = 'tb_ace_sw_1m', \
                           csv = csv, livedb = livedb)
    

    #convert to pandas DataFrame
    #MAYBE MOVE THIS STEP INTO THE GET DATA FUNCTION!!!!
    mag = pd.DataFrame(data = mag_data, columns = mag_data.dtype.names)
    sw = pd.DataFrame(data = sw_data, columns = sw_data.dtype.names)  
    sw.rename(columns={'dens': 'n', 'speed': 'v', 'temperature':'t'}, inplace=True)
        
    #clean data
    mag_clean, sw_clean = clean_data(mag, sw)

    #Create stucture to hold smoothed data
    smooth_num = 100
    
    col_names = ['date', 'bx', 'by', 'bz', 'bt', 'theta_z', 'theta_y']        
    data_mag = pd.concat([mag_clean['date'], \
            pd.Series(mag_clean['gsm_bx']).rolling(window = smooth_num).mean(), \
            pd.Series(mag_clean['gsm_by']).rolling(window = smooth_num).mean(),\
            pd.Series(mag_clean['gsm_bz']).rolling(window = smooth_num).mean(), \
            pd.Series(mag_clean['bt']).rolling(window = smooth_num).mean(),\
            pd.Series(mag_clean['gsm_lat']).rolling(window = smooth_num).mean(), \
            pd.Series(mag_clean['gsm_lon']).rolling(window = smooth_num).mean()], axis=1, keys = col_names)

        
    col_names_sw = ['date', 'sw_v', 'sw_n'] 
    data_sw = pd.concat([sw_clean['date'], \
        pd.Series(sw_clean['v']).rolling(window = smooth_num).mean(),\
        pd.Series(sw_clean['n']).rolling(window = smooth_num).mean()], axis=1, keys = col_names_sw )   

    data = pd.merge(data_mag, data_sw, how='left', left_on='date', right_on='date')



    #add empty columns for the predicted data values at each step in time
    data['istart_bz'] = 0
    data['iend_bz'] = 0
    data['tau_predicted'] = 0
    data['tau_actual'] = 0
    data['frac_est'] = 0
    data['bzm_predicted'] = 0
    data['bzm_actual'] = 0
    data['i_bzmax'] = 0
    data['theta_z_max'] = 0
    data['dtheta_z'] = 0
    
    data['istart_by'] = 0
    data['iend_by'] = 0
    data['tau_predicted_y'] = 0
    data['tau_actual_y'] = 0
    data['bym_predicted'] = 0
    data['bym_actual'] = 0
    data['i_bymax'] = 0
    data['theta_y_max'] = 0
    data['dtheta_y'] = 0
    
    data['lambda'] = 0
    data['chi'] = 0

    return data, data_mag, data_sw



