# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:13:45 2017

@author: hazel.bain
    This module investigates the adaptive kernel smoothing widths to be used 
    when smoothing data for input into the 
    Chen geoeffective magnetic cloud prediction Bayesian formulation. 
    Due to a relatively small smaple of geoeffective events, kernel density estimation
    is used to smooth the data and generate a non parametric PDFs.
    
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
    
"""

from sklearn.neighbors import KernelDensity
from scipy import stats
import pickle as pickle
import scipy.integrate as integrate
   
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def adaptive_test(events_frac, kernel_alg = 'scipy_stats', \
                ranges = [-150, 150, -250, 250], nbins = [50j, 100j],\
                ew = 2, nw = 0.5, ewc = 4, nwc = 0.5, plotting=[0,0,0,0,0], plottype = '2d'):

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
    
    import MC_predict_pdfs as mc
    
    kernel_alg = 'scipy_stats'
    #ranges = [-80, 80, -150, 150]
    nbins = [50j, 100j]
    #ew = 2; nw = 0.1
    
    Pbzm_tau_e, norm_bzm_tau_e = mc.P_bzm_tau_e(events_frac, ranges=ranges,\
        nbins=nbins, kernel_width = ewc, plotfig = 0)
    
    Pbzmp_taup_e, norm_bzmp_taup_e = mc.P_bzmp_taup_e(events_frac, ranges=ranges, \
        nbins=nbins, kernel_width = ewc, plotfig = 0)
    
    Pbzmp_taup_n, norm_bzmp_taup_n = mc.P_bzmp_taup_n(events_frac, ranges=ranges,\
        nbins=nbins, kernel_width = nwc, plotfig=0)
    
    print("max P_bzm_tau_e: "+str(Pbzm_tau_e.max())+", min w_bzm_tau_e: "+str(Pbzm_tau_e.min()))
    print("max P_bzmp_taup_e: "+str(Pbzmp_taup_e.max())+", min w_bzmp_taup_e: "+str(Pbzmp_taup_e.min()))
    print("max P_bzmp_taup_n: "+str(Pbzmp_taup_n.max())+", min w_bzmp_taup_n: "+str(Pbzmp_taup_n.min()))
    print("\n")
    
    #calculate the adaptive kernel smoothing wdith 
    w_bzm_tau_e = calc_width(Pbzm_tau_e, ew)
    w_bzmp_taup_e = calc_width(Pbzmp_taup_e, ew)
    w_bzmp_taup_n  = calc_width(Pbzmp_taup_n, nw)
    
    print("max w_bzm_tau_e: "+str(w_bzm_tau_e.max())+", min w_bzm_tau_e: "+str(w_bzm_tau_e.min()))
    print("max w_bzmp_taup_e: "+str(w_bzmp_taup_e.max())+", min w_bzmp_taup_e: "+str(w_bzmp_taup_e.min()))
    print("max w_bzmp_taup_n: "+str(w_bzmp_taup_n.max())+", min w_bzmp_taup_n: "+str(w_bzmp_taup_n.min()))
    print("\n")
               
    if plottype == '3d':
    
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
        X_bzm, Y_tau = np.mgrid[ranges[0]:ranges[1]:nbins[0], ranges[2]:ranges[3]:nbins[1]]
        dt0 = int(len(Y_tau[1])/2)
        b = X_bzm[:,0]
        t = Y_tau[0,dt0::]
    
        print(X_bzm[:, dt0::].shape)
        print(Y_tau[:,dt0::].shape)
        print(w_bzm_tau_e.shape)
    
        image = np.random.uniform(low = 0, high = 255,size=(200, 200))
        #w_bzm_tau_e = image
        #w_bzmp_taup_e[:,:,5] = image
        #w_bzmp_taup_n[:,:,5] = image
    
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R)
    
    
        fig.gca(projection='3d')
        c1 = ax1.plot_surface(X_bzm[:, dt0::],Y_tau[:,dt0::],w_bzm_tau_e,\
            cmap=plt.cm.coolwarm, linewidth=0, rstride = 2, cstride = 2)
        ax1.set_xlabel('Bzm')
        ax1.set_ylabel('Tau')
        ax1.set_title('P_bzm_tau_e, bandwidth = '+str(nw), fontsize='small')

    
        c2 = ax2.plot_surface(X_bzm[:, dt0::],Y_tau[:,dt0::],w_bzmp_taup_e[:,:,5],\
            cmap=plt.cm.gist_earth_r, linewidth=0, rstride = 2, cstride = 2)
        ax2.set_xlabel('Bzm')
        ax2.set_ylabel('Tau')
        ax2.set_title('P_bzmp_taup_e, bandwidth = '+str(nw), fontsize='small')
    
        c3 = ax3.plot_surface(X_bzm[:, dt0::],Y_tau[:,dt0::],w_bzmp_taup_n[:,:,5],\
            cmap=plt.cm.gist_earth_r, linewidth=0, rstride = 2, cstride = 2)
        ax3.set_xlabel('Bzm')
        ax3.set_ylabel('Tau')
        ax3.set_title('P_bzmp_taup_n, bandwidth = '+str(nw), fontsize='small')
    
    else:
        
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
        fontP = FontProperties() 
        fig.tight_layout()
        plt.subplots_adjust(wspace = 0.25)
        
        c1 = ax1.imshow(np.rot90(w_bzm_tau_e), extent=(ranges[0],ranges[1],0,ranges[3]), \
                        cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax1.set_xlim([ranges[0], ranges[1]])
        ax1.set_ylim([0, ranges[3]])
        ax1.set_xlabel('Bzm')
        ax1.set_ylabel('Tau')
        ax1.set_title('P_bzm_tau_e, bandwidth = '+str(ew), fontsize='small')
        fig.colorbar(c1, ax = ax1, fraction=0.025)
        #ax1.legend(loc='upper right', prop = fontP, fancybox=True)
        
        c2 = ax2.imshow(np.rot90(w_bzmp_taup_e[:,:,5]), extent=(ranges[0],ranges[1],0,ranges[3]), \
                        cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax2.set_xlim([ranges[0], ranges[1]])
        ax2.set_ylim([0, ranges[3]])
        ax2.set_xlabel('Bzm')
        ax2.set_ylabel('Tau')
        ax2.set_title('P_bzmp_taup_e, bandwidth = '+str(ew), fontsize='small')
        fig.colorbar(c2, ax = ax2, fraction=0.025)
        #ax2.legend(loc='upper right', prop = fontP, fancybox=True)
        
        c3 = ax3.imshow(np.rot90(w_bzmp_taup_n[:,:,5]), extent=(ranges[0],ranges[1],0,ranges[3]), \
                        cmap=plt.cm.gist_earth_r, interpolation = 'none')
        ax3.set_xlim([ranges[0], ranges[1]])
        ax3.set_ylim([0, ranges[3]])
        ax3.set_xlabel('Bzm')
        ax3.set_ylabel('Tau')
        ax3.set_title('P_bzmp_taup_n, bandwidth = '+str(nw), fontsize='small')
        fig.colorbar(c3, ax = ax3, fraction=0.025)
        #ax3.legend(loc='upper right', prop = fontP, fancybox=True)
    
    
    #create a dictionary to return PDFs etc
    P_dict = {}
    P_dict["P_bzm_tau_e"] = Pbzm_tau_e
    P_dict["norm_bzm_tau_e"] = norm_bzm_tau_e
    P_dict["w_bzm_tau_e"] = w_bzm_tau_e
    P_dict["P_bzmp_taup_e"] = Pbzmp_taup_e
    P_dict["norm_bzmp_taup_e"] = norm_bzmp_taup_e
    P_dict["w_bzmp_taup_e"] = w_bzmp_taup_e
    P_dict["P_bzmp_taup_n"] = Pbzmp_taup_n
    P_dict["norm_bzmp_taup_n"] = norm_bzmp_taup_n
    P_dict["w_bzmp_taup_n"] = w_bzmp_taup_n

    
    #save the input paramters as well
    P_dict["ew"] = ew
    P_dict["nw"] = nw
    P_dict["ranges"] = ranges
    P_dict["nbins"] = nbins
    P_dict["kernel_alg"] = kernel_alg
       
    #save a pickle file with P_dict
    #pickle.dump(open("Pdict_nw"+str(nw)+"_ew"+str(ew)+".p", "wb"))
    
    
    return P_dict    


def calc_width(pdf, w0, geoeff = 1):
    
    """
    determine the apadtive smoothing width for adaptive KDE based on fixed 
    KDE smoothing width
    
    inputs:
    -------
    
    pdf = array
        pdf array that has been smoothed with fixed kernel width kde
    w0 = float 
        fixed kernel smoothing width used to create pdf
    
    """
    
    from scipy.stats.mstats import gmean
    
    if geoeff == 1: 
        n = 51
    else:
        n = 656 
    
    
    if len(pdf.shape) == 2:
    
        wn0 = np.where(pdf > 0.0)
        #g = gmean(pdf.ravel())
        #g = np.exp( (1/n) * np.sum(np.log(pdf)))
        g = np.mean(pdf)
        w = w0 * (pdf / g)**-0.5

    else:
        w = np.zeros((pdf.shape))
        for i in range(pdf.shape[-1]):
            
            wn0 = np.where(pdf[:,:,i] > 0.0)
            #g = gmean(pdf[wn0[0],wn0[1],i].ravel())
            #g = np.exp( (1/n) * np.sum(np.log(pdf[wn0[0],wn0[1],i])))
            g = np.mean(pdf[wn0[0],wn0[1],i])
            w[:,:,i] = w0 * (pdf[:,:,i] / g)**-0.5

    return w
