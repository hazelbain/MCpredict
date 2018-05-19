#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:16:52 2018

@author: hazelbain
"""

    from sklearn.metrics import roc_curve
    import scikitplot as skplt

    events_time_frac = pickle.load(open("train/events_time_frac_fitall3_train_dst80_kp6_clean2.p","rb"))
    Pdict = pickle.load(open("PDFs/Pdict_30interp_100_75_2.p","rb"))

    
    events_time_frac = pgeo(events_time_frac, Pdict)
    
    
    #event indices for icme sheath, icme or icme-icme interactions that dont' have any MC 
    icme_evt_id = [1352,5562,4887,2721,2075,2732,432,4227,10894,713,3285,2597,11729,1441,1905]
    for id in icme_evt_id:
        criteria = 'evt_index == '+str(id)
        events_time_frac.loc[events_time_frac.eval(criteria), 'geoeff'] = 3
        
    #all icme into geoeff
    #criteria = 'geoeff == 3'
    #events_time_frac.loc[events_time_frac.eval(criteria), 'geoeff'] = 1
        
    
    e = events_time_frac.drop_duplicates(["evt_index"], keep = 'last')\
        [["evt_index","start","bzm","tau","bzm_predicted","tau_predicted","dst","bzmp_ind","taup_ind","geoeff","P1"]]
        
        
    w_geoeff = np.where(e['geoeff'] == 1.0)[0]
    w_no_geoeff = np.where(e['geoeff'] == 0)[0]
    w_reject = np.where(e['geoeff'] == 9)[0]
    w_bound = np.where(e['geoeff'] == 2)[0]
    w_icme = np.where(e['geoeff'] == 3)[0]
    w_test = np.where(e['geoeff'] == 999)[0]
    
    w_geoeff = np.where(np.logical_and(e['geoeff'] == 1.0, e['dst'] < -100))[0]
    w_no_geoeff = np.where(np.logical_and(e['geoeff'] == 0.0, e['dst'] > -100))[0]
    w_icme = np.where(np.logical_and(e['geoeff'] == 3.0, e['dst'] < -100))[0]
                  
    plt.scatter(events['bzm'].iloc[w_no_geoeff], e['tau'].iloc[w_no_geoeff], c = 'b', s=20, label = 'Not Geoeffecive' )   
    plt.scatter(events['bzm'].iloc[w_icme], events['tau'].iloc[w_icme], c = 'y', s=20, label = 'Geoeffective ICME')
    plt.scatter(events['bzm'].iloc[w_geoeff], e['tau'].iloc[w_geoeff], c = 'r', s=20, label = 'Geoeffecive MC' )                         
    #plt.scatter(events['bzm'].iloc[w_reject], events['tau'].iloc[w_reject], c = 'g', label = 'Reject')
    #plt.scatter(events['bzm'].iloc[w_bound], events['tau'].iloc[w_bound], c = 'orange', label = 'Bound')    
    plt.ylim(0,100)
    plt.xlim(-75,75)
    plt.xlabel("$\mathrm{B_{zm}}$ (nT)")
    plt.ylabel("Duration (hr)")
    leg = plt.legend(loc='upper left', fancybox=True, scatterpoints = 1 )
    plt.savefig('/Users/hazelbain/Dropbox/MCpredict/MCpredict/talkplots/geo_vs_nongeo_icme.pdf')


        
    #e.loc[e.index[np.logical_and(e['dst'] >= -100, e['geoeff'] == 1.0)]].geoeff = 0.0

    #ROC
    fpr, tpr, thresholds = roc_curve(e.geoeff, e.P1, pos_label = 1.0)
    #skplt.metrics.plot_roc_curve(e.geoeff.values, e.P1.values)
    
    #skill score
    tp=[]
    fp=[]
    fn=[]
    tn=[]
    for thresh in thresholds:

        tp.append(len(e.query("geoeff == 1.0 and P1 >= "+str(thresh))))
        tn.append(len(e.query("geoeff == 0.0 and P1 <= "+str(thresh))))
        fp.append(len(e.query("geoeff == 0.0 and P1 >= "+str(thresh))))
        fn.append(len(e.query("geoeff == 1.0 and P1 <= "+str(thresh))))
    
    fp = np.asarray(fp)
    tp = np.asarray(tp)
    tn = np.asarray(tn)
    fn = np.asarray(fn)
    
    fpr2 = fp / (fp + tn)
    tpr2= tp / (tp + fn)
    skill = tp / (tp + fp + fn)
    
    skill_max = skill.max()
    thresh_max = thresholds[np.argmax(skill)]
    thresh_ind = np.argmax(skill)
    
    #0.5
    #thresh_ind = np.min(np.where(thresholds < 0.5))
    #thresh_max = thresholds[thresh_ind]
    
    print("skill_max %f, thresh_max %f, thresh_ind %i" % (skill_max, thresh_max, thresh_ind))
    print("TP %i, FP %i, FN %i" % (tp[thresh_ind], fp[thresh_ind], fn[thresh_ind]))
    
    
    fp_events = e.query("geoeff == 0.0 and P1 >= "+str(thresh_max))
    fn_events = e.query("geoeff == 1.0 and P1 <= "+str(thresh_max))
    
    print("FP %i, fp_events %i, FN %i, fN_events %i " % (fp[thresh_ind], len(fp_events),fn[thresh_ind], len(fn_events)))    
    
    
    cnt=0
    for fnevind in fn_events.evt_index:
        
        tmpP1 = events_time_frac.query('evt_index == '+str(fnevind)).P1.values
        
        a = tmpP1
        b = np.zeros(len(a))

        for i in range(len(a)):
            if a[i] > 0.5:
                b[i] = b[i-1] + 1
            else:
                b[i] = 0
        if max(b) > 2:
            cnt+=1
        
    print((tp[thresh_ind]+cnt) / ((tp[thresh_ind]+cnt) + fp[thresh_ind] + (fn[thresh_ind]-cnt)))



    
    
    
    #nfp = 18
    
    
    
    tpb = len(e.query("geoeff == 1.0 and P1 >= "+str(thresh_max)))
    tnb = len(e.query("geoeff == 0.0 and P1 <= "+str(thresh_max)))
    fpb = len(e.query("geoeff == 0.0 and P1 >= "+str(thresh_max)))
    fnb = len(e.query("geoeff == 1.0 and P1 <= "+str(thresh_max)))
    tpb / (tpb + fpb + fnb)
    
    dst_thresh = -80
    thresh_max = 0.5
    tpd = len(e.query("dst <= "+str(dst_thresh)+" and P1 >= "+str(thresh_max)))
    tnd = len(e.query("dst > "+str(dst_thresh)+" and P1 <= "+str(thresh_max)))
    fpd = len(e.query("dst > "+str(dst_thresh)+" and P1 >= "+str(thresh_max)))
    fnd = len(e.query("dst <= "+str(dst_thresh)+" and P1 <= "+str(thresh_max)))
    tpd / (tpd + fpd + fnd)
    
    
    
    tp=[]
    fp=[]
    fn=[]
    tn=[]
    dst_thresh=-80
    for thresh in thresholds:

        tp.append(len(e.query("(geoeff == 1.0 or geoeff == 0.0) and dst <= "+str(dst_thresh)+" and P1 >= "+str(thresh))))
        tn.append(len(e.query("(geoeff == 1.0 or geoeff == 0.0) and dst > "+str(dst_thresh)+" and P1 <= "+str(thresh))))
        fp.append(len(e.query("(geoeff == 1.0 or geoeff == 0.0) and dst > "+str(dst_thresh)+" and P1 >= "+str(thresh))))
        fn.append(len(e.query("(geoeff == 1.0 or geoeff == 0.0)  and dst <= "+str(dst_thresh)+" and P1 <= "+str(thresh))))
    
    fp = np.asarray(fp)
    tp = np.asarray(tp)
    tn = np.asarray(tn)
    fn = np.asarray(fn)
    
    fpr2 = fp / (fp + tn)
    tpr2= tp / (tp + fn)
    skill = tp / (tp + fp + fn)
    
    skill_max = skill.max()
    thresh_max = thresholds[np.argmax(skill)]
    
    