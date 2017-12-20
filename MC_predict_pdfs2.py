#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:16:52 2017

@author: hazelbain
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:15:05 2017

@author: hazel.bain

    This module generates the PDFs for the
    Chen geoeffective magnetic cloud prediction Bayesian formulation. 
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    The data is stored as a pickle file and should be read in as:
    
        events_frac = pickle.load(open("events_frac.p","rb"))
    
    The top level create_pdfs function generates all the input PDFs
    and returns the posterior PDF P((Bzm, tau) n e|Bzm', tau' ; f) along with
    some other diagnostic variables. 
    
        Pbzm_tau_e_bzmp_taup, norm_bzm_tau_e_bzmp_taup, P0, P1, P1_map = create_pdfs(events_frac, kernel_alg = 'scipy_stats')
    
    Due to a relatively small smaple of geoeffective events, kernel density estimation
    is used to smooth the data and generate a non parametric PDFs.


"""


from sklearn.neighbors import KernelDensity
from scipy import stats
import pickle as pickle
import platform
import scipy.integrate as integrate
from scipy import interpolate
   
#import kde_hb as hb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D


def create_pdfs(events_time_frac, kernel_alg = 'scipy_stats', \
                ranges = [-150, 150, -250, 250], nbins = [50j, 100j],\
                fracs = [0.0, 0.2, 1.0], ew = 2, nw = 0.5, plotting=[0,0,0,0,0], fname='' ):

    """
    Create the PDFs for the
    Chen geoeffective magnetic cloud prediction Bayesian formulation. 
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
       
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
    kernel_alg = string
        choose between scikit learn and scipy stats python KDE algorithms
    ranges = 4 element array 
        defines the axis ranges for Bzm and tau [bmin, bmax, tmin, tmax]
    nbins = 2 elements array 
        defines the number of bins along bzm and tau [nbins_Bzm, nbins_tau]
    ew = float
        defines the kernel smoothing width for the geoeffective events
    nw = float
        defines the kernel smoothing width for the nongeoeffective events    
    plotting = int array
        indices indicate which PDFs to plot       
    
    """     
    
    if platform.system() == 'Darwin':
        proj_dir = '/Users/hazelbain/Dropbox/MCpredict/MCpredict/'
    
    
    #create input PDFS
    Pe = P_e(events_time_frac)
    Pn = P_n(events_time_frac)
    
    #Add a perturbation to Bzmp to break symmtry for KDE to work
    #b_indices = np.mgrid[ranges[0]:ranges[1]:nbins[0]]
    #delta_b = b_indices[1] - b_indices[0]
    #np.random.seed(2)
    #events_frac.bzm_predicted = events_frac.bzm_predicted + np.random.uniform(-delta_b, delta_b, len(events_frac))
    
    
    Pbzm_tau_e = P_bzm_tau_g(events_time_frac, g = 1, ranges=ranges,\
        nbins=nbins, kernel_width = ew, plotfig = plotting[0])
    
    Pbzm_tau_n = P_bzm_tau_g(events_time_frac, g = 0, ranges=ranges,\
        nbins=nbins, kernel_width = nw, plotfig = plotting[1])
    
    Pbzmp_taup_bzm_tau_e, indices = P_bzmp_taup_bzm_tau_g(events_time_frac, g = 1,\
        ranges=ranges, fracs=fracs, nbins=nbins, kernel_width = ew, plotfig=plotting[2])
    
    Pbzmp_taup_bzm_tau_n, indices = P_bzmp_taup_bzm_tau_g(events_time_frac, g = 0,\
        ranges=ranges, fracs=fracs, nbins=nbins, kernel_width = nw, plotfig=plotting[3])
        
    posterior, norm_posterior, norm, prob_e, prob_n, axis_vals = P_bzm_tau_g_bzmp_taup(Pe, \
                                                    Pn,\
                                                    Pbzm_tau_e, \
                                                    Pbzm_tau_n, \
                                                    Pbzmp_taup_bzm_tau_e, \
                                                    Pbzmp_taup_bzm_tau_n, \
                                                    ranges = ranges, nbins = nbins,\
                                                    fracs=fracs, \
                                                    plotfig = plotting[4])
    
    #record just the geoeffective posterior into the term P_bzm_tau_e_bzmp_taup 
    P_bzm_tau_e_bzmp_taup = norm_posterior[0,:,:,:,:,:] 
    
    #plot a record of the input and output PDFs
    plot_pdfs(events_time_frac, P_dict)
    
    #interpolate to finer grid 
    prob_e_interp, xnew, ynew = interpolate_to_fine_grid(P_dict)
    


    #create a dictionary to return PDFs etc
    P_dict = {}
    P_dict["P_e"] = Pe
    P_dict["P_n"] = Pn
    P_dict["P_bzm_tau_e"] = Pbzm_tau_e
    P_dict["P_bzm_tau_n"] = Pbzm_tau_n
    P_dict["P_bzmp_taup_bzm_tau_e"] = Pbzmp_taup_bzm_tau_e
    P_dict["P_bzmp_taup_bzm_tau_n"] = Pbzmp_taup_bzm_tau_n
    P_dict["P_bzm_tau_e_bzmp_taup"] = P_bzm_tau_e_bzmp_taup
    P_dict["posterior"] = posterior
    P_dict["norm_posterior"] = norm_posterior
    P_dict["norm"] = norm
    P_dict["prob_e"] = prob_e
    P_dict["prob_n"] = prob_n
    P_dict["prob_e_interp"] = prob_e_interp
    P_dict["indices"] = indices
    P_dict["axis_vals"] = axis_vals
    
    #save the input paramters as well
    P_dict["ew"] = ew
    P_dict["nw"] = nw
    P_dict["ranges"] = ranges
    P_dict["nbins"] = nbins
    P_dict["kernel_alg"] = kernel_alg
    P_dict["fracs"] = fracs
       
    #save a pickle file with P_dict
    pickle.dump(P_dict, open(proj_dir + "PDFs/Pdictn_"+fname+".p", "wb"))
    
    return P_dict    

def plot_pdfs(events_time_frac, P_dict):
    
    """
    Plot each of the input PDF files for the priors and likelihood terms.
    
    """
    
    if platform.system() == 'Darwin':
        proj_dir = '/Users/hazelbain/Dropbox/MCpredict/MCpredict/'
    
    indices = P_dict['indices']
    ranges = P_dict['ranges']
    nbins = P_dict['nbins']
    fracs = P_dict['fracs']
    
    ##plot all pdfs in output figure       
    for i in range(1, len(fracs)):
    
        gbzm = events_time_frac.query('geoeff == 1.0 and bzm < 0.0  and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).bzm
        gtau = events_time_frac.query('geoeff == 1.0 and bzm < 0.0  and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).tau
        
        gbzmn = events_time_frac.query('geoeff == 0.0 and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).bzm
        gtaun = events_time_frac.query('geoeff == 0.0 and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).tau
        
        gbzmp = events_time_frac.query('geoeff == 1.0 and bzm < 0.0  and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).bzm_predicted
        gtaup = events_time_frac.query('geoeff == 1.0 and bzm < 0.0  and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).tau_predicted
        gbzmp.dropna(inplace=True)
        gtaup.dropna(inplace=True)

        gbzmp_n = events_time_frac.query('geoeff == 0.0 and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).bzm_predicted
        gtaup_n = events_time_frac.query('geoeff == 0.0 and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).tau_predicted 
        gbzmp_n.dropna(inplace=True)
        gtaup_n.dropna(inplace=True)


        fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize = (8.5,11))
        fontP = FontProperties()
        fontP.set_size('x-small')
        
        c1 = ax1.imshow(np.rot90(Pbzm_tau_e), extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax1.plot(gbzm, gtau, 'k.', markersize=4, c='r', label = 'bzm, tau, g = 1')
        ax1.set_xlim([ranges[0], ranges[1]])
        ax1.set_ylim([0, ranges[3]])
        ax1.set_xlabel('Bzm')
        ax1.set_ylabel('Tau')
        ax1.set_title('P_bzm_tau_e, bandwidth = '+str(ew), fontsize = 'small')
        #cbaxes1 = ax1.add_axes([0.8, 0.1, 0.03, 0.8]) 
        #fig.colorbar(c1, ax = ax1, fraction=0.025, format='%.2E')      #, cax = cbaxes1)
        leg1 = ax1.legend(loc='upper right', prop = fontP, fancybox=True)
        leg1.get_frame().set_alpha(0.5)
        
        c2 = ax2.imshow(np.rot90(Pbzm_tau_n), extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax2.plot(gbzmn, gtaun, 'k.', markersize=4, c='b', label = 'bzm, tau, g = 0')
        ax2.set_xlim([ranges[0], ranges[1]])
        ax2.set_ylim([0, ranges[3]])
        ax2.set_xlabel('Bzm')
        ax2.set_ylabel('Tau')
        ax2.set_title('P_bzm_tau_n, bandwidth = '+str(nw), fontsize = 'small')
        #fig.colorbar(c2, ax = ax2, fraction=0.025, format='%.2E')
        leg2 = ax2.legend(loc='upper right', prop = fontP, fancybox=True)
        leg2.get_frame().set_alpha(0.5)
        
        #print(np.squeeze(indices).shape)
        predicted_duration = 15.0
        predicted_bzmax = -26.0
        indt = np.min(np.where(indices[3,:] > predicted_duration))
        indb = np.max(np.where(indices[2,:] < predicted_bzmax))
        
        c3 = ax3.imshow(np.rot90(Pbzmp_taup_bzm_tau_e[indb,indt,:,:,i-1]), extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax3.plot(gbzm, gtau, 'k.', markersize=4, c='r', label = 'bzm, tau, g = 1')
        ax3.set_xlim([ranges[0], ranges[1]])
        ax3.set_ylim([0, ranges[3]])
        ax3.set_xlabel('Bzm')
        ax3.set_ylabel('Tau')
        ax3.set_title('P_bzmp_taup_bzm_tau_e, bandwidth = '+str(ew), fontsize = 'small')
        #cbaxes3 = ax1.add_axes([0.8, 0.1, 0.03, 0.8]) 
        #fig.colorbar(c3, ax = ax5, fraction=0.025, format='%.2E')      #, cax = cbaxes3)
        leg3 = ax3.legend(loc='upper right', prop = fontP, fancybox=True)
        leg3.get_frame().set_alpha(0.5)
    
        c4 = ax4.imshow(np.rot90(Pbzmp_taup_bzm_tau_n[indb,indt,:,:,i-1]), extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax4.plot(gbzmn, gtaun, 'k.', markersize=4, c='b', label = 'bzm, tau, g = 0')
        #ax4.plot(gbzmp, gtaup, 'k.', markersize=4, c='r', label = 'bzmp, taup, g = 1')
        ax4.set_xlim([ranges[0], ranges[1]])
        ax4.set_ylim([0, ranges[3]])
        ax4.set_xlabel('Bzm')
        ax4.set_ylabel('Tau')
        ax4.set_title('P_bzmp_taup_bzm_tau_n, bandwidth = '+str(nw), fontsize = 'small')
        #fig.colorbar(c4, ax = ax4, fraction=0.025, format='%.2E')
        leg4 = ax4.legend(loc='upper right', prop = fontP, fancybox=True)
        leg4.get_frame().set_alpha(0.5)
        
        c5 = ax5.imshow(np.rot90(P_bzm_tau_e_bzmp_taup[:,:,indb,indt,i-1]), extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax5.plot(gbzmn, gtaun, 'k.', markersize=4, c='b', label = 'bzm, tau, g = 0')
        ax5.plot(gbzm, gtau, 'k.', markersize=4, c='r', label = 'bzm, tau, g = 1')
        ax5.set_xlim([ranges[0], ranges[1]])
        ax5.set_ylim([0, ranges[3]])
        ax5.set_xlabel('Bzm')
        ax5.set_ylabel('Tau')
        ax5.set_title('P_bzm_tau_e_bzmp_taup', fontsize = 'small')
        #cbaxes5 = ax5.add_axes([0.8, 0.1, 0.03, 0.8]) 
        #fig.colorbar(c5, ax = ax5, fraction=0.025, format='%2.2f')      #, cax = cbaxes5)
        leg5 = ax5.legend(loc='upper right', prop = fontP, fancybox=True)
        leg5.get_frame().set_alpha(0.5)
    
        c6 = ax6.imshow(np.rot90(prob_e[:,:,i-1]), extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax6.plot(gbzmn, gtaun, 'k.', markersize=4, c='b', label = 'bzm, tau, g = 0')
        ax6.plot(gbzm, gtau, 'k.', markersize=4, c='r', label = 'bzm, tau, g = 1')
        ax6.set_xlim([ranges[0], ranges[1]])
        ax6.set_ylim([0, ranges[3]])
        ax6.set_xlabel('Bzm_p')
        ax6.set_ylabel('Tau_p')
        ax6.set_title('Prob Geoeffective', fontsize = 'small')
        fig.colorbar(c6, ax = ax6, fraction=0.025, format='%2.2f')
        leg6 = ax6.legend(loc='upper right', prop = fontP, fancybox=True)
        leg6.get_frame().set_alpha(0.5)
    
        plt.tight_layout()
    
        fig.savefig(proj_dir + 'PDFs/plots/allpdfs_'+fname+'_f'+str(fracs[i])+'.pdf')
        plt.close()


def load_pdfs(fname=''):
    
    pp = pickle.load(open("PDFs/Pdict_"+fname+".p","rb"))
    
    return pp['P_e'], pp['P_n'], pp['P_bzm_tau_e'], pp['P_bzmp_taup_e'], pp['P_bzmp_taup_n'], pp['P_bzmp_taup_bzm_tau_e'], pp['P_bzm_tau_e_bzmp_taup']

    
def P_e(events_frac):  
    
    """
    Determine the prior PDF P(e) for input into the following Chen geoeffective 
    magnetic cloud prediction Bayesian formulation
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where c_j can be e or n
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
        as a function of the fraction of time through an event
            
    """
    
    #----P(e) - probability of a geoefective event 
    n_events = len(events_frac.drop_duplicates('evt_index'))
    n_geoeff_events = len(events_frac.drop_duplicates('evt_index').query('geoeff == 1.0'))
    
    #values from Chen paper
    #n_nongeoeff_events = 8600.
    #n_geoeff_events = 56.
    #n_events = n_nongeoeff_events + n_geoeff_events
    
    P_e = n_geoeff_events / n_events
    
    print('\n\n P_e: ' + str(P_e) + '\n\n')
    
    return P_e


def P_n(events_frac):  
    
    """
    Determine the prior PDF P(n) for input into the following Chen geoeffective 
    magnetic cloud prediction Bayesian formulation
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where c_j can be e or n
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
        as a function of the fraction of time through an event
            
    """
    
    #----P(n) probability of nongeoeffective events
    n_events = len(events_frac.drop_duplicates('evt_index'))
    n_nongeoeff_events = len(events_frac.drop_duplicates('evt_index').query('geoeff == 0.0'))
    
    #values from Chen paper
    #n_nongeoeff_events = 8600.
    #n_geoeff_events = 56.
    #n_events = n_nongeoeff_events + n_geoeff_events
    
    P_n = n_nongeoeff_events / n_events

    print('\n\n P_n: ' + str(P_n) + '\n\n')

    return P_n


def P_bzm_tau_g(events_frac, g = 1, kernel_alg = 'scipy_stats', \
                ranges = [-150, 150, -250, 250], nbins = [50j, 100j],\
                kernel_width = 2, plotfig = 0):  
    
    """
    Determine the prior PDF P(Bzm, tau|e), the probability of geoeffective event with observered bzm and tau
    for input into the following Chen geoeffective magnetic cloud prediction Bayesian formulation
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    Due to a relatively small smaple of geoeffective events, kernel density estimation
    is used to smooth the data and generate a non parametric PDF. An artifact of 
    the KDE method is that the PDF will be smoothed to produce probabilities 
    extending to negative values of tau. To prevent this effect at the boundary 
    the raw data values are reflected in the tau = 0 axis and then the KDE is applied, 
    producing a symmetric density estimates about the axis of reflection - see 
    Silverman 1998 section 2.10. The required output PDF is obtained by selecting array 
    elements corresponding to postive values of tau 
    
    f'(x) = 2f(x) for x >= 0 and f'(x) = 0 for x < 0
    
    As a result of this reflection, the input range for tau extends to negative 
    values and nbins_tau is double nbins_bzm. The output PDF will be for 
    tau = [0, tmax]
    
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
    kernel_alg = string
        choose between scikit learn and scipy stats python KDE algorithms        
    ranges = 4 element array 
        defines the axis ranges for Bzm and tau [bmin, bmax, tmin, tmax]
    nbins = 2 elements array 
        defines the number of bins along bzm and tau [nbins_Bzm, nbins_tau]
    kernel_widths = float
        defines the kernel smoothing width
    plotfit = int
        plot a figure of the distribution         
    
    """ 
    
    if g == 1:
        print('\n\n P_bzm_tau_e: Starting calculation \n\n')
    else:
        print('\n\n P_bzm_tau_n: Starting calculation \n\n')
    
    
    #range of Bzm and tau to define PDFs 
    bmin = ranges[0]
    bmax = ranges[1]
    tmin = ranges[2]
    tmax = ranges[3]
    
    #number of data bins in each dimension (dt takes into account the reflection)
    db = nbins[0]
    dt = nbins[1]
    
    #true boundary for tau and the corresponding index in the pdf array
    taumin = 0
    #dt0 = dt/2 
        
    #extract raw data points from dataframe of "actual" bzm and tau for geoeffective events
    gbzm = events_frac.drop_duplicates('evt_index').query('geoeff == '+str(g)+' and bzm < 0.0').bzm
    gtau = events_frac.drop_duplicates('evt_index').query('geoeff == '+str(g)+' and bzm < 0.0 ').tau
    
    #print("max bzm: "+str(gbzm.max())+", tau: "+ str(gtau.max()))
    
    
    #to handle boundary conditions and limit the density estimate to positve 
    #values of tau: reflect the data points along the tau axis, perform the 
    #density estimate and then set negative values of tau to 0 and double the 
    #density of positive values of tau
    gbzm_r = np.concatenate([gbzm, gbzm]) 
    gtau_r = np.concatenate([gtau, -gtau])
    
    #grid containing x, y positions 
    X_bzm, Y_tau = np.mgrid[bmin:bmax:db, tmin:tmax:dt]
    dt0 = int(len(Y_tau[1])/2.)
    Y_tau = Y_tau-((tmax-tmin)/len(Y_tau[1])/2.)      #to make sure the resulting PDF tau 
                                            #axis will start at 0. 

    #option to use scikit learn or scipy stats kernel algorithms
    if kernel_alg == 'sklearn':
        
        positions = np.vstack([X_bzm.ravel(), Y_tau.ravel()]).T
        values = np.vstack([gbzm_r, gtau_r]).T
        kernel_bzm_tau_g = KernelDensity(kernel='gaussian', bandwidth=kernel_width).fit(values)
        Ptmp_bzm_tau_g = np.exp(np.reshape(kernel_bzm_tau_g.score_samples(positions).T, X_bzm.shape))
        
    elif kernel_alg == 'scipy_stats':
        
        positions = np.vstack([X_bzm.ravel(), Y_tau.ravel()])
        values = np.vstack([gbzm_r, gtau_r])
        kernel_bzm_tau_g = stats.gaussian_kde(values, bw_method = kernel_width)
        Ptmp_bzm_tau_g = np.reshape(kernel_bzm_tau_g(positions).T, X_bzm.shape)

    #set the density estimate to 0 for negative tau, and x2 for positve tau 
    P_bzm_tau_g = Ptmp_bzm_tau_g[:,dt0::]*2


    if plotfig == 1:
        
        fontP = FontProperties()                #legend
        #fontP.set_size('medium')1
        
        fig, ax = plt.subplots()
        c = ax.imshow(np.rot90(P_bzm_tau_g), extent=(bmin,bmax,taumin,tmax), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax.plot(gbzm, gtau, 'k.', markersize=4, label = 'bzm, tau, geoeff = 1')
        ax.set_xlim([bmin, bmax])
        ax.set_ylim([taumin, tmax])
        ax.set_xlabel('Bzm')
        ax.set_ylabel('Tau')
        ax.set_title('P_bzm_tau_e, bandwidth = '+str(ew))
        fig.colorbar(c)
        ax.legend(loc='upper right', prop = fontP, fancybox=True)


    return P_bzm_tau_g  



def P_bzmp_taup_bzm_tau_g(events_frac, g = 1,  kernel_alg = 'scipy_stats', \
                ranges = [-150, 150, -250, 250], nbins = [50j, 100j],\
                fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], kernel_width = 2, plotfig = 0):  
    
    """
    Determine the prior PDF P(Bzm', tau'|(Bzm, tau) n e ; f), the probability 
    of a geoeffective event with estimates Bzm' and tau' for a MC with actual 
    values Bzm and tau, at fraction f throughout an event for input into the following 
    Chen geoeffective magnetic cloud prediction Bayesian formulation. This is 
    the bayesian likelihood PDF, relating the model to the data.  
    
    P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
    
    Due to a relatively small smaple of geoeffective events, kernel density estimation
    is used to smooth the data and generate a non parametric PDF. An artifact of 
    the KDE method is that the PDF will be smoothed to produce probabilities 
    extending to negative values of tau. To prevent this effect at the boundary 
    the raw data values are reflected in the tau' = 0 axis and then the KDE is applied, 
    producing a symmetric density estimates about the axis of reflection - see 
    Silverman 1998 section 2.10. The required output PDF is obtained by selecting array 
    elements corresponding to postive values of tau 
    
    f'(x) = 2f(x) for x >= 0 and f'(x) = 0 for x < 0
    
    As a result of this reflection, the input range for tau extends to negative 
    values and nbins_tau is double nbins_bzm. The output PDF will be for 
    tau' = [0, tmax]
    
    inputs:
        
    events_frac = dataframe
        contains output variables from a fit to solar wind magnetic field data
    kernel_alg = string
        choose between scikit learn and scipy stats python KDE algorithms
    ranges = 4 element array 
        defines the axis ranges for Bzm and tau [bmin, bmax, tmin, tmax]
    nbins = 2 elements array 
        defines the number of bins along bzm and tau [nbins_Bzm, nbins_tau]
    fracs = list
        fractions to split the events into for calculating evolution dependent pdfs
    kernel_width = float
        defines the kernel smoothing width
    plotfit = int
        plot a figure of the distribution 
        
    
    """ 
    
    if g == 1:
        print('\n\n P_bzmp_taup_bzm_tau_e: Starting calculation \n\n')
    else:
        print('\n\n P_bzmp_taup_bzm_tau_n: Starting calculation \n\n')
    
    #range of Bzm and tau to define PDFs 
    bmin = ranges[0]
    bmax = ranges[1]
    tmin = ranges[2]
    tmax = ranges[3]
    
    #number of data bins in each dimension (dt takes into account the reflection)
    db = nbins[0]
    dt = nbins[1]
    
    #true boundary for tau and the corresponding index in the pdf array
    taumin = 0
    #dt0 = dt/2 
    
    
    #set grid containing x,y positions and use to get array size
    X_bzmp, Y_taup, XX_bzm, YY_tau = np.mgrid[bmin:bmax:db, tmin:tmax:dt, bmin:bmax:db, tmin:tmax:dt]
    db2 = int(len(X_bzmp[:,0,:,:]))
    dt2 = int(len(Y_taup[0,:,:,:]))
    
    #P_bzmp_taup_bzm_tau_e is a function of the fraction of time f throughout an event
    #currently the fit to the data considers every 5th of an event 
    nfracs = len(fracs)-1
    Ptmp_bzmp_taup_bzm_tau_g = np.zeros((db2,dt2,db2,dt2,nfracs))
    for i in range(1, len(fracs)):
        
        #extract raw data points from dataframe of estimates bzm and tau for 
        #fraction f throughout eoeffective events
        gbzm = events_frac.query('geoeff == '+str(g)+' and bzm < 0.0  and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).bzm
        gtau = events_frac.query('geoeff == '+str(g)+' and bzm < 0.0  and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).tau
        
        #extract raw data points from dataframe of estimates bzm' and tau' for 
        #fraction f throughout eoeffective events
        gbzmp = events_frac.query('geoeff == '+str(g)+' and bzm < 0.0  and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).bzm_predicted
        gtaup = events_frac.query('geoeff == '+str(g)+' and bzm < 0.0  and frac_est >= ' + str(fracs[i-1]) + ' and frac_est < ' + str(fracs[i])).tau_predicted
        
        if gtaup.max() == np.inf:
            inf_idx = gtaup[gtaup == np.inf].index.values[0]
            gtaup.drop([inf_idx], inplace = True)
            gbzmp.drop([inf_idx], inplace = True)
            gtau.drop([inf_idx], inplace = True)
            gbzm.drop([inf_idx], inplace = True)

    
        #to handle boundary conditions and limit the density estimate to positve 
        #values of tau' and tau: reflect the data points along the tau' and tau
        #axes, perform the ensity estimate and then set negative values of 
        #tau' and tau to 0 and x4 the density of positive values of tau'
        gbzmp_r = np.concatenate([gbzmp, gbzmp, gbzmp, gbzmp])
        gtaup_r = np.concatenate([gtaup, -gtaup, gtaup, -gtaup])        
        gbzm_r = np.concatenate([gbzm, gbzm, gbzm, gbzm]) 
        gtau_r = np.concatenate([gtau, gtau, -gtau, -gtau])
        
        #make sure the resulting PDF tau axis will start at 0 hours. 
        dt0 = int(len(Y_taup[1])/2)
        Y_taup = Y_taup-((tmax-tmin)/len(Y_taup[1])/2.)
        YY_tau = YY_tau-((tmax-tmin)/len(YY_tau[1])/2.)
        
        #option to use scikit learn or scipy stats kernel algorithms
        if kernel_alg == 'sklearn':
            
            positions = np.vstack([X_bzmp.ravel(), Y_taup.ravel(), XX_bzm.ravel(), YY_tau.ravel()]).T
            values = np.vstack([gbzmp_r, gtaup_r, gbzm_r, gtau_r]).T        
            kernel_bzmp_taup_bzm_tau_g = KernelDensity(kernel='gaussian', bandwidth=kernel_width).fit(values)
            
            Ptmp_bzmp_taup_bzm_tau_g[:,:,:,:,i-1] = np.exp(np.reshape(kernel_bzmp_taup_bzm_tau_g.score_samples(positions).T, X_bzmp.shape))
            
        elif kernel_alg == 'scipy_stats':    
            
            positions = np.vstack([X_bzmp.ravel(), Y_taup.ravel(), XX_bzm.ravel(), YY_tau.ravel()])
            values = np.vstack([gbzmp_r, gtaup_r, gbzm_r, gtau_r])  
            
            kernel_bzmp_taup_bzm_tau_g = stats.gaussian_kde(values, bw_method = kernel_width)
            Ptmp_bzmp_taup_bzm_tau_g[:,:,:,:,i-1] = np.reshape(kernel_bzmp_taup_bzm_tau_g(positions).T, X_bzmp.shape)

            
 
    #set the density estimate to 0 for negative tau, and x4 for positve tau 
    P_bzmp_taup_bzm_tau_g = Ptmp_bzmp_taup_bzm_tau_g[:,dt0::,:,dt0::,:]*4       #*50*50*2    

    #check the normalization of the 4D space   
    bp = X_bzmp[:,0,0,0]
    tp = Y_taup[0,dt0::,0,0]
    b = XX_bzm[0,0,:,0]
    t = YY_tau[0,0,0,dt0::]
    
    indices = np.squeeze(np.array([[bp],[tp],[b],[t]]))
    
                                
    if plotfig == 1:
        
        fontP = FontProperties()                #legend
        #fontP.set_size('medium')1
        
        fig, ax = plt.subplots()
        c = ax.imshow(np.rot90(P_bzmp_taup_bzm_tau_g[:,:,indb,indt,0]), extent=(bmin,bmax,taumin,tmax), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax.plot(gbzmp_n, gtaup_n, 'k.', markersize=4, c='b')
        ax.plot(gbzmp, gtaup, 'k.', markersize=4, c='r')
        ax.set_xlim([bmin, bmax])
        ax.set_ylim([taumin, tmax])
        ax.set_xlabel('Bzm')
        ax.set_ylabel('Tau')
        fig.colorbar(c)

    return P_bzmp_taup_bzm_tau_g, indices


def P_bzm_tau_g_bzmp_taup(P_e, P_n, P_bzm_tau_e, P_bzm_tau_n, P_bzmp_taup_bzm_tau_e,\
                P_bzmp_taup_bzm_tau_n, ranges = [-150, 150, -250, 250], \
                nbins = [50j, 100j], fracs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], plotfig = 0):  
    
    """
    Determine the posterior PDF P((Bzm, tau) n e |Bzm', tau' ; f), the probability 
    of a geoeffective event with parameters Bzm and tau given estimates  
    Bzm' and tau', at fraction f throughout an event for input into the following 
    Chen geoeffective magnetic cloud prediction Bayesian formulation. This is 
    the bayesian posterior PDF.
    
   P((Bzm, tau) n e|Bzm', tau' ; f) 
    = P(Bzm, tau|(Bzm, tau) n e ; f) * P(Bzm, tau|e) * P(e) / SUM_j( P(Bzm',tau' | c_j ; f) * P(c_j))
    
    where
    
    Bzm = actual value of Bz max for a magnetic cloud
    tau = actual duation of a magnetic cloud
    Bzm' = fitted/estimated value of Bz max at fraction (f) of an event
    tau' = fitted/estimated value of duration at fraction (f) of an event
    e = geoeffective event
    n = nongeoeffective event
    f = fraction of an event
       
    inputs:
    
    P_e = float
        Prior P(e)
    P_n = float
        Prior P(n)
    P_bzm_tau_e = data array
        Prior PDF P(bzm, tau|e)
    P_bzmp_taup_e = data array
        Evidence PDF P(bzm', tau'|e; f)
    P_bzmp_taup_n = data array
        Evidence PDF P(bzm', tau'|n; f)
    P_bzmp_taup_bzm_tau_e = data array
        Likelihood PDF P(Bzm', tau' | (Bzm, tau) n e ; f)

    ranges = 4 element array 
        defines the axis ranges for Bzm and tau [bmin, bmax, tmin, tmax]
    nbins = 2 elements array 
        defines the number of bins along bzm and tau [nbins_Bzm, nbins_tau]
    fracs = list
        fractions to split the events into for calculating evolution dependent pdfs
    plotfit = int
        plot a figure of the distribution     

    
    output:
        
    P_bzm_tau_e_bzmp_taup = data array
        Posterior PDF P((Bzm, tau) n e | Bzm', tau' ; f)
    
        
    """ 

    #range of Bzm and tau to define PDFs 
    bmin = ranges[0]
    bmax = ranges[1]
    tmin = ranges[2]
    tmax = ranges[3]
    
    #number of data bins in each dimension (dt takes into account the reflection)
    db = nbins[0]
    dt = nbins[1]
    
    #set grid containing x,y positions and use to get array size
    X_bzmp, Y_taup, XX_bzm, YY_tau = np.mgrid[bmin:bmax:db, tmin:tmax:dt, bmin:bmax:db, tmin:tmax:dt] 
    dt0 = int(len(Y_taup[1])/2.)
    db2 = int(len(X_bzmp[:,0,:,:]))
    dt2 = int(len(Y_taup[0,:,:,:])/2)
    
    
    #check the normalization of the 4D space   
    bp = X_bzmp[:,0,0,0]
    tp = Y_taup[0,dt0::,0,0]
    b = XX_bzm[0,0,:,0]
    t = YY_tau[0,0,0,dt0::]
    
    axis_vals = np.array([bp,tp,b,t])    
    
    #true boundary for tau and the corresponding index in the pdf array
    taumin = 0
    #dt0 = dt/2 
    
    #P_bzm_tau_e_bzmp_taup_e is a function of the fraction of time f throughout an event
    nfracs = len(fracs)-1    

    posterior = np.zeros((2, db2, dt2, db2, dt2, nfracs))
    norm_posterior = np.zeros((2, db2, dt2, db2, dt2, nfracs))
    norm = np.zeros((db2, dt2, nfracs))
    prob_e = np.zeros((db2, dt2, nfracs))
    prob_n = np.zeros((db2, dt2, nfracs))
    
    for i in range(nfracs):
        for j in range(db2):
            for k in range(dt2):
                    
                #bayes for the geoeffective events        
                bayes_e = np.multiply(P_bzmp_taup_bzm_tau_e[j,k,:,:,i], P_bzm_tau_e) * P_e
            
                #bayes for the non geoeffective events
                bayes_n = np.multiply(P_bzmp_taup_bzm_tau_n[j,k,:,:,i], P_bzm_tau_n) * P_n
    
                posterior[:,:,:,j,k,i] = np.array([bayes_e, bayes_n])
    
                #normalization value 
                norm_e = integrate.simps(integrate.simps(bayes_e, bp), tp)
                norm_n = integrate.simps(integrate.simps(bayes_n, bp), tp)
                norm[j,k,i] = norm_e + norm_n     
                    
                #normalize the posterior
                norm_posterior[:,:,:,j,k,i] = posterior[:,:,:,j,k,i] / norm[j,k,i]
                    
                #calculate the probability map for P1
                prob_e[j,k,i] = integrate.simps(integrate.simps(norm_posterior[0,:,:,j,k,i], bp), tp)
                prob_n[j,k,i] = integrate.simps(integrate.simps(norm_posterior[1,:,:,j,k,i], bp), tp)
                
            
    if plotfig == 1:
        
        fontP = FontProperties()                #legend
        #fontP.set_size('medium')1        
            
        fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,5))
    
        c1 = ax1.imshow(np.rot90(prob_e), extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax1.plot(gbzmn, gtaun, 'k.', markersize=4, c='b', label = 'bzm, tau, g = 1')
        ax1.plot(gbzm, gtau, 'k.', markersize=4, c='r', label = 'bzm, tau, g = 1')
        ax1.set_xlim([ranges[0], ranges[1]])
        ax1.set_ylim([0, ranges[3]])
        ax1.set_xlabel('Bzm')
        ax1.set_ylabel('Tau')
        ax1.set_title('Prob Geoeffective')
        fig.colorbar(c1, ax = ax1, fraction=0.025)
        
        c2 = ax2.imshow(np.rot90(prob_n), extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax2.plot(gbzmn, gtaun, 'k.', markersize=4, c='b', label = 'bzm, tau, g = 1')
        ax2.plot(gbzm, gtau, 'k.', markersize=4, c='r', label = 'bzm, tau, g = 1')
        ax2.set_xlim([ranges[0], ranges[1]])
        ax2.set_ylim([0, ranges[3]])
        ax2.set_xlabel('Bzm')
        ax2.set_ylabel('Tau')
        ax2.set_title('Prob Non Geoeffective')
        fig.colorbar(c2, ax = ax2, fraction=0.025)
        
        #fig.savefig('/Users/hazelbain/Dropbox/MCpredict/MCpredict/PDFs/plots/P1_bayes_test.pdf')
    

    return posterior, norm_posterior, norm, prob_e, prob_n, axis_vals 


def interpolate_to_fine_grid(P_dict, events_time_frac = events_time_frac, plot = 0, fracs = [0.2,1.0]):
    
    """
    Interpolate the P1 map to a finer grid scale to reduce compute time
    
    """

    if platform.system() == 'Darwin':
        proj_dir = '/Users/hazelbain/Dropbox/MCpredict/MCpredict/'
    
    indices = P_dict['indices']
    ranges = P_dict['ranges']
    nbins = P_dict['nbins']
    #fracs = P_dict['fracs']

    ##recreate the coarse grid points
    X_bzm, Y_tau = np.mgrid[ranges[0]:ranges[1]:nbins[0], \
                            ranges[2]:ranges[3]:nbins[1]]
    
    #number of data bins in each dimension (dt takes into account the reflection)
    db = int(len(X_bzm[:,0]))
    dt = int(len(Y_tau[0,:]))
    
    #only take the mesh for tau > 0 
    X_bzm = X_bzm[:,db:dt]
    Y_tau = Y_tau[:,db:dt]
    
    #prep the axes and grid for the interpolation
    x = np.arange(X_bzm[0,0], X_bzm[-1,0], X_bzm[1,0]-X_bzm[0,0])
    y = np.arange(Y_tau[0][0], Y_tau[0][-1], Y_tau[0][1]-Y_tau[0][0])
    xx, yy = np.meshgrid(x, y)
    z = np.rot90(P_dict['prob_e'][:,:,0])
    
    #interpolate
    f = interpolate.interp2d(x, y, z, kind='cubic')
    
    #create the new axis values
    nbins_new = [x*2 for x in nbins]
    X_bzm_new, Y_tau_new = np.mgrid[ranges[0]:ranges[1]:nbins_new[0], \
                            ranges[2]:ranges[3]:nbins_new[1]]
    
    X_bzm_new = X_bzm_new[:,db*2:dt*2]
    Y_tau_new = Y_tau_new[:,db*2:dt*2]
    
    xnew = np.arange(X_bzm_new[0,0], X_bzm_new[-1,0], X_bzm_new[1,0]-X_bzm_new[0,0])
    ynew = np.arange(Y_tau_new[0][0], Y_tau_new[0][-1], Y_tau_new[0][1]-Y_tau_new[0][0])
    ynew = np.append(ynew, [200.0])
    
    #create the new interpoated distribution
    znew = f(xnew, ynew)
    
    #renormalize 
    znew_norm = (znew - znew.min())/(znew.max()-znew.min())
    

    #plot
    if plot == 1:
    
        fontP = FontProperties()                #legend
        #fontP.set_size('medium')1        
            
        fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,5))
        
        gbzm = events_time_frac.query('geoeff == 1.0 and bzm < 0.0  and frac_est >= 0.6')['bzm']
        gtau = events_time_frac.query('geoeff == 1.0 and bzm < 0.0  and frac_est >= 0.6')['tau']
        
        gbzmn = events_time_frac.query('geoeff == 0.0 and frac_est >= 0.6 ' )['bzm']
        gtaun = events_time_frac.query('geoeff == 0.0 and frac_est >= 0.6 ' )['tau']
        
    
        c1 = ax1.imshow(np.rot90(P_dict['prob_e'][:,:,0]), extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax1.plot(gbzmn, gtaun, 'k.', markersize=4, c='b', label = 'bzm, tau, g = 1')
        ax1.plot(gbzm, gtau, 'k.', markersize=4, c='r', label = 'bzm, tau, g = 1')
        ax1.set_xlim([ranges[0], ranges[1]])
        ax1.set_ylim([0, ranges[3]])
        ax1.set_xlabel('Bzm (nT)')
        ax1.set_ylabel('Tau (hours)')
        ax1.set_title('Prob Geoeffective (coarse orig)')
        fig.colorbar(c1, ax = ax1, fraction=0.025)
        
        c2 = ax2.imshow(znew_norm, extent=(ranges[0],ranges[1],0,ranges[3]), cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax2.plot(gbzmn, gtaun, 'k.', markersize=4, c='b', label = 'bzm, tau, g = 1')
        ax2.plot(gbzm, gtau, 'k.', markersize=4, c='r', label = 'bzm, tau, g = 1')
        ax2.set_xlim([ranges[0], ranges[1]])
        ax2.set_ylim([0, ranges[3]])
        ax2.set_xlabel('Bzm (nT)')
        ax2.set_ylabel('Tau  (hours)')
        ax2.set_title('Prob Geoeffective (fine interp)')
        fig.colorbar(c2, ax = ax2, fraction=0.025)
    
        plt.tight_layout()
    
        fig.savefig(proj_dir + 'PDFs/plots/P1_bayes_interp_test_dst100.pdf')
    
    
    return znew_norm, xnew, ynew

