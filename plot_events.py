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
    events_frac = pickle.load(open(proj_dir + "train/events_frac_fitall_train_dst80_kp6.0.p","rb"))
    events = pickle.load(open(proj_dir + "train/events_fitall_train_dst80_kp6.0.p","rb"))
    
    #read in the dst data
    dst_data = dst.read_dst_df()
        
    #read in kp data
    kp_data = kp.read_kp()
    
    #read in Richardson and Cane ICME list 
    icme_list = icme.read_richardson_icme_list()
    
    #### step 1: gather events to use for the bayseian PDF, uses Chen_MC_prediction without predict keyword

    #t1 = ['1-jan-1998','1-jan-1999','1-jan-2000','1-jan-2001','1-jan-2002','1-jan-2003', '1-jan-2004','1-jan-2005',\
    #      '1-jan-2006','1-jan-2007','1-jan-2008','1-jan-2009','1-jan-2010','1-jan-2011','1-jan-2012','1-jan-2013',\
    #       '1-jan-2014','1-jan-2015','1-jan-2016','1-jan-2017']
    #t1 = ['29-jul-2001']

    
    #t2 = ['31-dec-1998','31-dec-1999','31-dec-2000','31-dec-2001','31-dec-2002','31-dec-2003', '31-dec-2004','31-dec-2005',\
    #      '31-dec-2006','31-dec-2007','31-dec-2008','31-dec-2009','31-dec-2010','31-dec-2011','31-dec-2012','31-dec-2013', \
    #       '31-dec-2014','31-dec-2015','31-dec-2016','31-may-2017']
    #t2 = ['12-aug-2001']
    
    #t1 = ['1-jan-1999']
    #t2 = ['31-jan-1999']
    
    
    #loop through the years
    for j in range(len(t1)):
            
        #format times
        start_date = datetime.strptime(t1[j], "%d-%b-%Y")
        end_date= datetime.strptime(t2[j], "%d-%b-%Y")
        
        #get the weekly start and end dates dates
        date_list = []
        cal = calendar.Calendar()
        for y in (np.arange(end_date.month - start_date.month + 1)+start_date.month):
            for x in cal.monthdatescalendar(start_date.year, y):
                date_list.append([x[0], x[0]+timedelta(days = 8)])
        date_list = np.asarray(date_list)
 
        print(date_list)
        
        #loop through the weeks
        for i in range(0,len(date_list)):
                    
            #format times
            st = datetime.combine(date_list[i,0], datetime.min.time())
            et = datetime.combine(date_list[i,1], datetime.min.time())
            
            if st >= start_date - timedelta(days = 7) and st <= end_date:
                
                #format time strings
                stf = datetime.strftime(st, "%Y-%m-%d")
                etf = datetime.strftime(et, "%Y-%m-%d")
                
                print("Start date: " + stf )
                print("End date  : " + etf + "\n")
                
                try:  

                    #get the ACE data
                    data, data_mag, data_sw = get_data_for_plots(stf, etf)
                    
                    #subset of events and events_frac
                    events_subset = events.query('start >= "'+stf+'" and end <= "'+ datetime.strftime(et + timedelta(days = 1), "%Y-%m-%d") + '"')
                    events_frac_subset = events_frac.query('start >= "'+stf+'" and end <= "'+ datetime.strftime(et + timedelta(days = 1), "%Y-%m-%d") + '"')
                    
                    events_subset.drop_duplicates(['start'], inplace = True)
                    events_frac_subset.drop_duplicates(['start','frac'], inplace = True)
                    
                    #start and end of event times
                    evt_times = events_subset[['start','end']].values
                    #uniq_index = np.unique(evt_times[:,0],return_index = True)[1]
                    #evt_times = evt_times[uniq_index, :]
                    
                    #start and end of ICME
                    line1 = list(icme_list.query('plasma_start >="'+stf+'" and plasma_start <= "'+ datetime.strftime(et + timedelta(days = 1), "%Y-%m-%d") +'"')\
                                [['plasma_start','plasma_end']].values.flatten())
                    
                    #start and end of MC
                    line2 = list(icme_list.query('mc_start >="'+stf+'" and mc_start <= "'+ datetime.strftime(et + timedelta(days = 1), "%Y-%m-%d") +'"')\
                                [['mc_start','mc_end']].values.flatten())
                    
                    #make the plot
                    mcpredict_plot(data, events_frac_subset, \
                                   dst_data[st - timedelta(1):et + timedelta(1)], \
                                   kp_data[st - timedelta(1):et + timedelta(1)], \
                                   line=line1, line2=line2, bars = evt_times, \
                                   plt_outfile = 'mcpredict_'+ datetime.strftime(date_list[i][0], "%Y-%m-%d_%H%M") + '.pdf',\
                                   plt_outpath = proj_dir + 'longterm_fitall_sw/')

                except:
                    print("something wrong")
                    
    #return data, data_mag, data_sw
    return None

def mcpredict_plot(data, events_frac, dst_data, kp_data, line=[], line2=[], bars = [], plot_fit = 1, dst_thresh = -80, kp_thresh = 6, \
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
    from matplotlib.dates import HourLocator
    from matplotlib.dates import DateFormatter
    from matplotlib.ticker import MultipleLocator
            
    #start and end times for plot to make sure all plots are consistent
    #st = datetime.strptime(data['date'][0]), "%Y-%m-%d")
    #et = datetime.strptime(data['date'][-1], "%Y-%m-%d")

    st = data['date'][0]
    et = data['date'].iloc[-1]
    
    #read in the dst data
    #dst_data = dst.read_dst(str(st), str(et))

    #plot the ace data
    f, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(7, figsize=(12,10))
 
    plt.subplots_adjust(hspace = .1)       # no vertical space between subplots
    fontP = FontProperties()                #legend
    fontP.set_size('medium')
    
    dateFmt = DateFormatter('%d-%b')
    hoursLoc = HourLocator()
    daysLoc = DayLocator()
    
    minorLocator = MultipleLocator(1)
    
    
    color = {0.0:'green', 1.0:'red', 2.0:'grey',3.0:'orange'}
    #fitcolor = {0.2:'purple', 0.4:'blue', events_frac.frac.iloc[3]:'green',0.8:'orange', 1.0:'red'}

    #----Bx
    ax0.plot(data['date'], data['bx'], label='Bx (nT)')
    ax0.hlines(0.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax0.set_xticklabels(' ')
    ax0.xaxis.set_major_locator(daysLoc)
    ax0.xaxis.set_minor_locator(hoursLoc)
    ax0.set_xlim([st, et])
    ax0.yaxis.set_minor_locator(minorLocator)
    for l in line:
        ax0.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax0.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    for b in range(len(bars)):
        ax0.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15)        
    leg = ax0.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)

  
    #----By
    ax1.plot(data['date'], data['by'], label='By (nT)')
    ax1.hlines(0.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax1.set_xticklabels(' ')
    ax1.xaxis.set_major_locator(daysLoc)
    ax1.xaxis.set_minor_locator(hoursLoc)
    ax1.set_xlim([st, et])
    ax1.yaxis.set_minor_locator(minorLocator)
    for l in line:
        ax1.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax1.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    for b in range(len(bars)):
        ax1.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15)        
    leg = ax1.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)

        
    #----Bz
    ax2.plot(data['date'], data['bz'], label='Bz (nT)')
    ax2.hlines(0.0, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax2.set_xticklabels(' ')
    ax2.xaxis.set_major_locator(daysLoc)
    ax2.xaxis.set_minor_locator(hoursLoc)
    ax2.set_xlim([st, et])
    ax2.yaxis.set_minor_locator(minorLocator)
    for l in line:
        ax2.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax2.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    for b in range(len(bars)):
        ax2.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
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
# 
#     #max bz line
#     for b in range(len(bars)):
#         if events_frac['geoeff'].iloc[b*6] == 1.0:
#             ax1.hlines(events_frac['bzm'].iloc[b*6], bars[b,0], bars[b,1], linestyle='-',color='grey')
# 
#     #plot the fitted profile at certain intervals through the event  
#     if plot_fit == 1:
#         for i in range(len(events_frac)): 
#             #only plot the fits for the geoeffective events
#             if (events_frac['geoeff'].iloc[i] == 1.0) & (events_frac['frac'].iloc[i] >0.1):
#                  
#                 #for each fraction of an event, determine the current fit to the profile up to this point
#                 pred_dur = events_frac['tau_predicted'].iloc[i] * 60.
#                 fit_times = [ events_frac['start'].iloc[i] + timedelta(seconds = j*60) for j in np.arange(pred_dur)]
#                 fit_profile = events_frac['bzm_predicted'].iloc[i] * np.sin(np.pi*np.arange(0,1,1./(pred_dur)) )          
#                 
#                 ax1.plot(fit_times, fit_profile, color=fitcolor[events_frac['frac'].iloc[i]])    
#==============================================================================

    #----density
    ax3.plot(data['date'], data['sw_n'], label='n ($\mathrm{cm^-3}$)')
    ax3.set_xticklabels(' ')
    ax3.xaxis.set_major_locator(daysLoc)
    ax3.xaxis.set_minor_locator(hoursLoc)
    ax3.set_xlim([st, et])
    for l in line:
        ax3.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax3.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    for b in range(len(bars)):
        ax3.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    leg = ax3.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    #----velocity
    maxv = max(  data['sw_v'].loc[np.where(np.isnan(data['sw_v']) == False )] ) + 50
    minv =  min(  data['sw_v'].loc[np.where(np.isnan(data['sw_v']) == False )] ) - 50
    ax4.plot(data['date'], data['sw_v'], label='v ($\mathrm{km s^-1}$)')
    ax4.set_ylim(top = maxv, bottom = minv)
    ax4.set_xticklabels(' ')
    ax4.xaxis.set_major_locator(daysLoc)
    ax4.xaxis.set_minor_locator(hoursLoc)
    ax4.set_xlim([st, et])
    for l in line:
        ax4.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax4.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    for b in range(len(bars)):
        ax4.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15)       
    leg = ax4.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
            
    #----dst
    ax5.plot(dst_data[st:et].index, dst_data[st:et]['dst'], label='Dst')
    ax5.hlines(dst_thresh, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax5.set_xticklabels(' ')
    ax5.set_xlim([st, et])
    for l in line:
        ax5.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax5.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    for b in range(len(bars)):
        ax5.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['dstgeoeff'].iloc[b*6]], alpha=0.15) 
    #ax7.set_xlabel("Start Time "+ str(st)+" (UTC)")
    leg = ax5.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)
    
    
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

    ax6.hlines(kp_thresh, data['date'][0], data['date'].iloc[-1], linestyle='--',color='grey')
    ax6.set_xticklabels(' ')
    ax6.xaxis.set_major_formatter(dateFmt)
    ax6.xaxis.set_major_locator(daysLoc)
    ax6.xaxis.set_minor_locator(hoursLoc)
    ax6.set_xlim([st, et])
    ax6.set_ylim(0,10)
    for l in line:
        ax6.axvline(x=l, linewidth=2, linestyle='--', color='black')
    for l2 in line2:
        ax6.axvline(x=l2, linewidth=2, linestyle=':', color='red')
    for b in range(len(bars)):
        ax6.axvspan(bars[b,0], bars[b,1], facecolor=color[events_frac['geoeff'].iloc[b*6]], alpha=0.15) 
    ax6.set_xlabel("Start Time "+ str(st)+" (UTC)")
    leg = ax6.legend(loc='upper left', prop = fontP, fancybox=True, frameon=False )
    leg.get_frame().set_alpha(0.5)        


    plt.savefig(plt_outpath + plt_outfile, format='pdf')

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



