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


def adaptive_test(events_frac, kernel_alg = 'scipy_stats', \
                ranges = [-150, 150, -250, 250], nbins = [50j, 100j],\
                ew = 2, nw = 0.5, plotting=[0,0,0,0,0]):

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
    
    Pbzm_tau_e, norm_bzm_tau_e = P_bzm_tau_e(events_frac, ranges=ranges,\
        nbins=nbins, kernel_width = ew, plotfig = plotting[0])
    
    Pbzmp_taup_e, norm_bzmp_taup_e = P_bzmp_taup_e(events_frac, ranges=ranges, \
        nbins=nbins, kernel_width = ew, plotfig = plotting[1])
    
    Pbzmp_taup_n, norm_bzmp_taup_n = P_bzmp_taup_n(events_frac, ranges=ranges,\
        nbins=nbins, kernel_width = nw, plotfig=plotting[2])
    
    
    #create a dictionary to return PDFs etc
    P_dict = {}
    P_dict["P_bzm_tau_e"] = Pbzm_tau_e
    P_dict["norm_bzm_tau_e"] = norm_bzm_tau_e
    P_dict["P_bzmp_taup_e"] = Pbzmp_taup_e
    P_dict["norm_bzmp_taup_e"] = norm_bzmp_taup_e
    P_dict["P_bzmp_taup_n"] = Pbzmp_taup_n
    P_dict["norm_bzmp_taup_n"] = norm_bzmp_taup_n

    
    #save the input paramters as well
    P_dict["ew"] = ew
    P_dict["nw"] = nw
    P_dict["ranges"] = ranges
    P_dict["nbins"] = nbins
    P_dict["kernel_alg"] = kernel_alg
       
    #save a pickle file with P_dict
    pickle.dump(open("Pdict_nw"+str(nw)+"_ew"+str(ew)+".p", "wb"))
    
    return P_dict    

def calc_width(pdf, w0):
    
    """"
    determine the apadtive smoothing width for adaptive KDE based on fixed 
    KDE smoothing width
    
    inputs:
    -------
    
    pdf = array
        pdf array that has been smoothed with fixed kernel width kde
    w0 = float 
        fixed kernel smoothing width used to create pdf
    
    """"
    
