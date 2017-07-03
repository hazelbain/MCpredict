# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:16:33 2017

@author: hazel.bain
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plot_predict_bz_tau_frac(events_frac, outname = 'bztau_predict.pdf'):
    
    """
    
    Plots the fitted magnetic cloud predicted Bzm vs predicted duration tau as
    a function of the fraction of the event. 
    
    input
    
    events: dataframe
        dataframe containing events determined from historical data as a fraction
        of the event 
    
    """
    
    
    from matplotlib.font_manager import FontProperties
        
    ##plot Bzm vs tau
    w2 = np.where((events_frac['frac'] == 0.2) & (events_frac['geoeff'] == 1.0))[0]
    w4 = np.where((events_frac['frac'] == 0.4) & (events_frac['geoeff'] == 1.0))[0]
    w6 = np.where((events_frac['frac'] == events_frac.frac.iloc[3]) & (events_frac['geoeff'] == 1.0))[0]
    w8 = np.where((events_frac['frac'] == 0.8) & (events_frac['geoeff'] == 1.0))[0]
    w10 = np.where((events_frac['frac'] == 1.0) & (events_frac['geoeff'] == 1.0))[0]
    
    bt = events_frac['bzm'].iloc[w2]*events_frac['tau'].iloc[w2]  
                                 
    bt2_predict = events_frac['bzm_predicted'].iloc[w2]*events_frac['tau_predicted'].iloc[w2]  
    bt4_predict = events_frac['bzm_predicted'].iloc[w4]*events_frac['tau_predicted'].iloc[w4]  
    bt6_predict = events_frac['bzm_predicted'].iloc[w6]*events_frac['tau_predicted'].iloc[w6]  
    bt8_predict = events_frac['bzm_predicted'].iloc[w8]*events_frac['tau_predicted'].iloc[w8] 
    bt10_predict = events_frac['bzm_predicted'].iloc[w10]*events_frac['tau_predicted'].iloc[w10] 


    fontP = FontProperties()                #legend
    fontP.set_size('medium')                       
                           
    plt.scatter(bt, bt2_predict, c = 'purple', label = '0.2 event')
    plt.scatter(bt, bt4_predict, c = 'b', label = '0.4 event')
    plt.scatter(bt, bt6_predict, c = 'g', label = '0.6 event')
    plt.scatter(bt, bt8_predict, c = 'orange', label = '0.8 event')  
    plt.scatter(bt, bt8_predict, c = 'r', label = '1.0 event')                         
    
    plt.plot(bt, bt, c = 'black')      
    
    #plt.ylim(0,60)
    #plt.xlim(-60,60)
    plt.title("BzmTau obs vs predicted as fraction \n of event duration (Geoeff = 1)")
    plt.xlabel("$\mathrm{B_{zm} tau (obs)}$")
    plt.ylabel("$\mathrm{B_{zm} tau (predict)}$")
    leg = plt.legend(loc='upper left', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg.get_frame().set_alpha(0.5)
    
    #plt.show()
    plt.savefig(outname, format='pdf')
    
    plt.close()
    
    return None
    
def plot_obs_bz_tau(events, outname = 'bzm_vs_tau.pdf'):
    
    """
    Plots the magnetic cloud actual bzm vs tau
    
    input
    
    events: dataframe
        dataframe containing events determined from historical data
    
    
    """
    
    from matplotlib.font_manager import FontProperties
        
    ##plot Bzm vs tau
    w_geoeff = np.where(events['geoeff'] == 1.0)[0]
    w_no_geoeff = np.where(events['geoeff'] == 0)[0]

                           
    fontP = FontProperties()                #legend
    fontP.set_size('medium')                       
                           
                           
    plt.scatter(events['bzm'].iloc[w_no_geoeff], events['tau'].iloc[w_no_geoeff], c = 'b', label = 'Not Geoeffecive' )                         
    plt.scatter(events['bzm'].iloc[w_geoeff], events['tau'].iloc[w_geoeff], c = 'r', label = 'Geoeffective')
    plt.ylim(0,60)
    plt.xlim(-60,60)
    plt.xlabel("$\mathrm{B_{zm}}$ (nT)")
    plt.ylabel("Duration (hr)")
    leg = plt.legend(loc='upper right', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg.get_frame().set_alpha(0.5)
    
    plt.savefig(outname, format='pdf')
    
    plt.close()


def plot_obs_bz_tau_dst(events, outname = 'bzm_vs_tau_vs_dst', fname = ''):
    
    """
    Plots the magnetic cloud actual bzm vs tau as a function of dst
    
    input
    
    events: dataframe
        dataframe containing events determined from historical data
    
    
    """
    
    from matplotlib.font_manager import FontProperties
        
    ##plot Bzm vs tau
    w_geoeff = np.where(events['geoeff'] == 1.0)[0]
    w_no_geoeff = np.where(events['geoeff'] == 0)[0]

    w_no_ambig = np.where(events['geoeff'] < 2.0)[0]
                           
    fontP = FontProperties()                #legend
    fontP.set_size('medium')                       
                           
    c = plt.scatter(events['bzm'].iloc[w_no_ambig], events['tau'].iloc[w_no_ambig], c = events['dst'].iloc[w_no_ambig])                       
    plt.ylim(0,100)
    plt.xlim(-60,60)
    plt.xlabel("$\mathrm{B_{zm}}$ (nT)")
    plt.ylabel("Duration (hr)")
    plt.title("Chen events: Bzm vs tau vs Dst (no ambig)")
    leg = plt.legend(loc='upper right', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg.get_frame().set_alpha(0.5)
    cbar = plt.colorbar(c, label = "Dst")
    #cbar.set_label("Dst")   
    
    plt.savefig(outname + '_' + fname + '.pdf', format='pdf')
    
    plt.close()    
   
    
def plot_bzm_vs_tau_skill(events_frac, outname = 'bzm_vs_tau_skill', fname = ''):
    
    """
    Plots the magnetic cloud actual bzm vs tau for each fraction of an 
    event. Plots missed and false alarms as well
    
    input
    
    events_frac: dataframe
        dataframe containing events determined from historical data
    
    
    """
     
    corpos = events_frac.query('geoeff == 1.0 and frac == 1.0 and P1_scaled > 0.2').sort_values(by='start')[['bzm','tau','dst']]
    corneg = events_frac.query('geoeff == 0.0 and frac == 1.0 and P1_scaled < 0.2').sort_values(by='start')[['bzm','tau','dst']]
    missed = events_frac.query('geoeff == 1.0 and frac == 1.0 and P1_scaled < 0.2').sort_values(by='start')[['bzm','tau','dst']]
    false = events_frac.query('geoeff == 0.0 and frac == 1.0 and P1_scaled > 0.2').sort_values(by='start')[['bzm','tau','dst']]
                           
    fontP = FontProperties()                #legend
    fontP.set_size('medium')                       
                                               
    plt.scatter(corneg['bzm'], corneg['tau'], c = 'b', label = 'CorNeg' )                         
    plt.scatter(corpos['bzm'], corpos['tau'], c = 'r', label = 'CorPos')
    plt.scatter(missed['bzm'], missed['tau'], c = 'g', label = 'Missed' )                         
    plt.scatter(false['bzm'], false['tau'], c = 'orange', label = 'False')
    plt.ylim(0,100)
    plt.xlim(-60,60)
    plt.xlabel("$\mathrm{B_{zm}}$ (nT)")
    plt.ylabel("Duration (hr)")
    leg = plt.legend(loc='upper right', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg.get_frame().set_alpha(0.5)
    
    plt.savefig(outname + '_' + fname + '.pdf', format='pdf')
    
    plt.close()
    
    