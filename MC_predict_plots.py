# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:16:33 2017

@author: hazel.bain
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plot_predict_bz_tau_frac(events_frac, outname = 'valid/plots/bztau_predict', fname = ''):
    
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


    fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2, figsize = (10,15))

    fontP = FontProperties()                #legend
    fontP.set_size('medium')                       
                           
    ax0.scatter(bt, bt2_predict, c = 'purple', label = '0.2 event')
    ax0.plot(bt, bt, c = 'black') 
    ax0.set_ylim(-2000,1000)
    ax0.set_xlim(-900,200)
    #ax0.set_title("BzmTau obs vs predicted as fraction \n of event duration (Geoeff = 1)")
    ax0.set_xlabel("$\mathrm{B_{zm} tau (obs)}$")
    ax0.set_ylabel("$\mathrm{B_{zm} tau (predict)}$")
    leg0 = ax0.legend(loc='upper left', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg0.get_frame().set_alpha(0.5)
    
    ax1.scatter(bt, bt4_predict, c = 'b', label = '0.4 event')
    ax1.plot(bt, bt, c = 'black') 
    ax1.set_ylim(-2000,1000)
    ax1.set_xlim(-900,200)
    #ax1.set_title("BzmTau obs vs predicted as fraction \n of event duration (Geoeff = 1)")
    ax1.set_xlabel("$\mathrm{B_{zm} tau (obs)}$")
    ax1.set_ylabel("$\mathrm{B_{zm} tau (predict)}$")
    leg1 = ax1.legend(loc='upper left', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg1.get_frame().set_alpha(0.5)
    
    ax2.scatter(bt, bt6_predict, c = 'g', label = '0.6 event')
    ax2.plot(bt, bt, c = 'black')  
    ax2.set_ylim(-2000,1000)
    ax2.set_xlim(-900,200)
    #ax2.set_title("BzmTau obs vs predicted as fraction \n of event duration (Geoeff = 1)")
    ax2.set_xlabel("$\mathrm{B_{zm} tau (obs)}$")
    ax2.set_ylabel("$\mathrm{B_{zm} tau (predict)}$")
    leg2 = ax2.legend(loc='upper left', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg2.get_frame().set_alpha(0.5)    
    
    ax3.scatter(bt, bt8_predict, c = 'orange', label = '0.8 event')  
    ax3.plot(bt, bt, c = 'black') 
    ax3.set_ylim(-2000,1000)
    ax3.set_xlim(-900,200)
    #ax3.set_title("BzmTau obs vs predicted as fraction \n of event duration (Geoeff = 1)")
    ax3.set_xlabel("$\mathrm{B_{zm} tau (obs)}$")
    ax3.set_ylabel("$\mathrm{B_{zm} tau (predict)}$")
    leg3 = ax3.legend(loc='upper left', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg3.get_frame().set_alpha(0.5) 
    
    ax4.scatter(bt, bt8_predict, c = 'r', label = '1.0 event')        
    ax4.plot(bt, bt, c = 'black') 
    ax4.set_ylim(-2000,1000)
    ax4.set_xlim(-900,200)
    #ax4.set_title("BzmTau obs vs predicted as fraction \n of event duration (Geoeff = 1)")
    ax4.set_xlabel("$\mathrm{B_{zm} tau (obs)}$")
    ax4.set_ylabel("$\mathrm{B_{zm} tau (predict)}$")
    leg4 = ax4.legend(loc='upper left', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg4.get_frame().set_alpha(0.5)                  
    
    ax5.scatter(bt, bt10_predict, c = 'r', label = '1.0 event')        
    ax5.plot(bt, bt, c = 'black') 
    ax5.set_ylim(-2000,1000)
    ax5.set_xlim(-900,200)
    #ax4.set_title("BzmTau obs vs predicted as fraction \n of event duration (Geoeff = 1)")
    ax5.set_xlabel("$\mathrm{B_{zm} tau (obs)}$")
    ax5.set_ylabel("$\mathrm{B_{zm} tau (predict)}$")
    leg5 = ax5.legend(loc='upper left', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg5.get_frame().set_alpha(0.5)  
    
    #plt.show()
    plt.savefig(outname + '_' + fname + '.jpeg', format='jpeg')
    
    plt.close()
    
    return None
    
def plot_obs_bz_tau(events, outname = 'valid/plots/bzm_vs_tau', fname=''):
    
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
    
    plt.savefig(outname + '_' + fname + '.jpeg', format='jpeg')
    
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
    
    plt.savefig(outname + '_' + fname + '.jpeg', format='jpeg')
    
    plt.close()    
   
    
def plot_bzm_vs_tau_skill(events_frac, P1 = 0.2, outname = 'valid/plots/bzm_vs_tau_skill', fname = ''):
    
    """
    Plots the magnetic cloud actual bzm vs tau for each fraction of an 
    event. Plots missed and false alarms as well
    
    input
    
    events_frac: dataframe
        dataframe containing events determined from historical data
    
    
    """
     
    corpos = events_frac.query('geoeff == 1.0 and frac == 1.0 and P1_scaled >' + str(P1)).sort_values(by='start')[['bzm','tau','dst']]
    corneg = events_frac.query('geoeff == 0.0 and frac == 1.0 and P1_scaled <'  + str(P1)).sort_values(by='start')[['bzm','tau','dst']]
    missed = events_frac.query('geoeff == 1.0 and frac == 1.0 and P1_scaled <'  + str(P1)).sort_values(by='start')[['bzm','tau','dst']]
    false = events_frac.query('geoeff == 0.0 and frac == 1.0 and P1_scaled > ' + str(P1)).sort_values(by='start')[['bzm','tau','dst']]
    
    corpos_p = events_frac.query('geoeff == 1.0 and frac == 1.0 and P1_scaled >' + str(P1)).sort_values(by='start')[['bzm_predicted','tau_predicted','dst']]
    corneg_p = events_frac.query('geoeff == 0.0 and frac == 1.0 and P1_scaled <'  + str(P1)).sort_values(by='start')[['bzm_predicted','tau_predicted','dst']]
    missed_p = events_frac.query('geoeff == 1.0 and frac == 1.0 and P1_scaled <'  + str(P1)).sort_values(by='start')[['bzm_predicted','tau_predicted','dst']]
    false_p = events_frac.query('geoeff == 0.0 and frac == 1.0 and P1_scaled > ' + str(P1)).sort_values(by='start')[['bzm_predicted','tau_predicted','dst']]
    
    CSI = len(corpos) / (len(corpos)+len(false)+len(missed))
                       
    fontP = FontProperties()                #legend
    fontP.set_size('x-small')                       
    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (10,5))
    

    ##plots with actual values                                       
    ax0.scatter(corneg['bzm'], corneg['tau'], c = 'b', label = 'CorNeg' )                         
    ax0.scatter(corpos['bzm'], corpos['tau'], c = 'r', label = 'CorPos')
    ax0.scatter(missed['bzm'], missed['tau'], c = 'g', label = 'Missed' )                         
    ax0.scatter(false['bzm'], false['tau'], c = 'orange', label = 'False')
    ax0.set_title("Observed parameters")
    ax0.set_ylim(0,100)
    ax0.set_xlim(-60,60)
    ax0.set_xlabel("$\mathrm{B_{zm}}$ (nT)")
    ax0.set_ylabel("Duration (hr)")
    leg0 = ax0.legend(loc='upper right', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg0.get_frame().set_alpha(0.5)
    
    #add a table with the skill scores
    cell_text = [[str(len(corneg)), str(len(false))],\
                 [str(len(missed)), str(len(corpos))]]
    
    table0 = ax0.table(cellText = cell_text, \
              cellColours=[['b','orange'],['g','r']],\
              rowLabels=['NG','G'], \
              colLabels=['NG','G'],\
              loc='center left',\
              colLoc='center', rowLoc='center',\
              bbox = [0.075, 0.75, 0.25, 0.2])
    table0.scale(0.25, 0.5)
    table0.set_fontsize(10)
    
    ax0.annotate("CSI = %.2f" % CSI, xy=(0.1,0.96), xycoords='axes fraction', color='Black', fontsize=10)
    
    #plots with predicted values
    ax1.scatter(corneg_p['bzm_predicted'], corneg_p['tau_predicted'], c = 'b', label = 'CorNeg' )                         
    ax1.scatter(corpos_p['bzm_predicted'], corpos_p['tau_predicted'], c = 'r', label = 'CorPos')
    ax1.scatter(missed_p['bzm_predicted'], missed_p['tau_predicted'], c = 'g', label = 'Missed' )                         
    ax1.scatter(false_p['bzm_predicted'], false_p['tau_predicted'], c = 'orange', label = 'False')
    ax1.set_title("Predicted parameters")
    ax1.set_ylim(0,100)
    ax1.set_xlim(-60,60)
    ax1.set_xlabel("$\mathrm{B_{zm}}$ predicted (nT)")
    ax1.set_ylabel("Duration predicted (hr)")
    leg1 = ax1.legend(loc='upper right', prop = fontP, fancybox=True, \
                     frameon=True, scatterpoints = 1 )
    leg1.get_frame().set_alpha(0.5)
   
    plt.savefig(outname + '_' + fname + '.jpeg', format='jpeg')
    
    plt.close()
    
    
def plot_obs_vs_predict(events_frac, outname = 'valid/plots/bzm_obs_vs_predicted', fname = ''):
     
    from matplotlib.font_manager import FontProperties
        
    
    evts = events_frac.query('geoeff == 1.0 and frac == 1.0')[['bzm','tau','bzm_predicted','tau_predicted']]
    
                           
    fontP = FontProperties()                #legend
    fontP.set_size('medium')                       
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                       
    ax1.scatter(evts['bzm'], evts['bzm_predicted'])                       
    ax1.set_ylim(-60,60)
    ax1.set_xlim(-60,60)
    ax1.set_xlabel("$\mathrm{B_{zm}}$ obs (nT)")
    ax1.set_ylabel("$\mathrm{B_{zm}}$ obs (nT)")
    ax1.set_title("Bzm obs vs predicted")
    
    ax2.scatter(evts['tau'], evts['tau_predicted'])                       
    ax2.set_ylim(0,100)
    ax2.set_xlim(0,100)
    ax2.set_xlabel("tau obs (nT)")
    ax2.set_ylabel("tau obs (nT)")
    ax2.set_title("Tau obs vs predicted")

    plt.savefig(outname + '_' + fname + '.jpeg', format='jpeg')
    
    plt.close()  
    
    return None

def plot_theta(events_frac, fname = ''):
    
    evts = events_frac[['bzm','bzm_predicted','tau','tau_predicted','frac','dtheta_z','theta_z_max']]\
                        .iloc[np.where((events_frac.geoeff == 1))[0]]
    
    ax = evts.iloc[np.where(evts.theta_z_max < 0.0)[0]].boxplot(column = 'dtheta_z', by='frac')
    fig = ax.get_figure()
    fig.savefig('valid/plots/dtheta_'+fname+'.jpeg', format = 'jpeg')
    plt.close('all')
    
    #plot max theta
    ax = evts.iloc[np.where(evts.theta_z_max < 0.0)[0]].boxplot(column = 'theta_z_max', return_type='axes')
    fig = ax.get_figure()
    fig.savefig('valid/plots/theta_max_'+fname+'.jpeg', format = 'jpeg')
    plt.close('all')

def plot_boxplot(events_frac,fname=''):
    
    #boxplot
    #w = np.where(events_frac.frac == 1.0)[0]
    ax = events_frac.boxplot(column = 'P1_scaled', by = 'geoeff')
    fig = ax.get_figure()
    fig.savefig('valid/plots/P1_boxplot_'+fname+'.jpeg', format = 'jpeg')
    plt.close('all')
    

def write_report(events_frac, outname = 'html/mc_predict_test_results', fname = '', P1 = 0.2):
    
    
    ##make the plots   
    plot_predict_bz_tau_frac(events_frac, fname = fname)
    plot_obs_vs_predict(events_frac, fname = fname)
    plot_bzm_vs_tau_skill(events_frac, P1 = P1, fname = fname)
    plot_theta(events_frac, fname = fname)
    
    #missed, false = sort_incorrect(events_frac, fname = fname)
    
    #skill
#==============================================================================
#     corpos = events_frac.query('geoeff == 1.0 and frac == 1.0 and P1_scaled >' + str(P1)).sort_values(by='start')[['bzm','tau','dst']]
#     corneg = events_frac.query('geoeff == 0.0 and frac == 1.0 and P1_scaled <'  + str(P1)).sort_values(by='start')[['bzm','tau','dst']]
#     missed = events_frac.query('geoeff == 1.0 and frac == 1.0 and P1_scaled <'  + str(P1)).sort_values(by='start')[['bzm','tau','dst']]
#     false = events_frac.query('geoeff == 0.0 and frac == 1.0 and P1_scaled > ' + str(P1)).sort_values(by='start')[['bzm','tau','dst']]
#     
#     CSI = len(corpos) / (len(corpos)+len(false)+len(missed))
#==============================================================================
    
    
    ##open the html file
    f = open(outname + '_' + fname + '.html', 'w')
     
    f.write('<!DOCTYPE html>\n')
    f.write('<html>\n')
    f.write('<head>\n')
    f.write('<title>MC predict Test Results '+fname+'</title>\n')
    f.write('</head>\n')
    f.write('<body>\n')
    
    #skill score
#==============================================================================
#     f.write('   <p>Skill Score %.2f </p>\n' % CSI )
#==============================================================================
    
    #contingency table
#==============================================================================
#     f.write('   <table border="1">\n')
#     f.write('   <caption> Contingency Table </caption>\n')
#     f.write('   <thead>')
#     f.write('   <tr>')
#     f.write('       <th scope="col">Obs\Pred</th>\n')
#     f.write('       <th scope="col">Nongeoeff</th>\n')
#     f.write('       <th scope="col">Geoeff</th>\n')
#     f.write('   </tr>\n')   
#     f.write('   </thead>\n')
#     f.write('   <tbody>\n')
#     f.write('   <tr>\n')
#     f.write('       <th scope="row">Nongeoeff</th>\n')
#     f.write('       <td>'+ str(len(corneg)) +'</td>\n')
#     f.write('       <td>'+ str(len(false)) +'</td>\n')
#     f.write('   </tr>\n')  
#     f.write('   <tr>\n')
#     f.write('       <th scope="row">Geoeff</th>\n')
#     f.write('       <td>'+ str(len(missed)) +'</td>\n')
#     f.write('       <td>'+ str(len(corpos)) +'</td>\n')
#     f.write('   </tr>\n') 
#     f.write('   </tbody>\n')     
#     f.write('  </table>\n')
#     
#==============================================================================
    
    #images 
    f.write('  <img src="valid/plots/bzm_vs_tau_skill' + '_' + fname + '.jpeg" alt="bzm_vs_tau_skill">\n\n\n')
    f.write('  <img src="valid/plots/bzm_obs_vs_predicted' + '_' + fname + '.jpeg" alt="bzm_obs_vs_predicted" ">\n\n\n')
    f.write('  <img src="valid/plots/bztau_predict' + '_' + fname + '.jpeg" alt="bztau_predict">\n\n\n')
    f.write('  <img src="valid/plots/dtheta_' + fname + '.jpeg" alt="dtheta">\n\n\n')
    f.write('  <img src="valid/plots/theta_max_' + fname + '.jpeg" alt="theta_max">\n\n\n')

    f.write('</body>')
    f.write('</html>')

    f.close()   
    
    #return missed, false
    return None
    
    
    
def sort_incorrect(events_frac, fname = ''):
    
    import shutil
    import calendar
    import datetime as datetime
    import os
    
    #missed = events_frac_predict2.query('geoeff == 1.0 and frac == 1.0 and P1_scaled < 0.2').sort_values(by='dst',ascending=1)[['start','dst','P1_scaled']]
    #false = events_frac_predict2.query('geoeff == 0.0 and frac == 1.0 and P1_scaled > 0.2').sort_values(by='dst',ascending=1)[['start','dst','P1_scaled']]

    missed = events_frac.query('geoeff == 1.0 and frac == 1.0 and P1_scaled < 0.2').sort_values(by='start')[['start','dst','P1_scaled']]
    false = events_frac.query('geoeff == 0.0 and frac == 1.0 and P1_scaled > 0.2').sort_values(by='start')[['start','dst','P1_scaled']]
    
    dd_longterm = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/longterm_th2/'
    dd_missed = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/missed_'+fname+'/'
    dd_false = 'C:/Users/hazel.bain/Documents/MC_predict/pyMCpredict/MCpredict/false_'+fname+'/'
    
    os.makedirs(dd_missed)
    os.makedirs(dd_false)
    
    
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
        
        shutil.copyfile( false_str, new_loc) 

    return missed, false    
    