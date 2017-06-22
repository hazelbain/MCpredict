# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:30:20 2017

@author: hazel.bain
"""

    ew = 4

    bmin = -75
    bmax = 75
    tmin = 0
    tmax = 75
    db = 50j
    dt = 50j

    #----P(Bzm, tau|e) -- probability of geoeffective event with observered bzm and tau
    gbzm = events_frac.bzm.iloc[np.where((events_frac.frac == 1.0) & (events_frac.geoeff == 1.0) & (events_frac.bzm < 0.0))[0]]
    gtau = events_frac.tau.iloc[np.where((events_frac.frac == 1.0) & (events_frac.geoeff == 1.0) & (events_frac.bzm < 0.0))[0]]

    #gbzm = events_frac.bzm.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff != 1.0))[0]]
    #gtau = events_frac.tau.iloc[np.where((events_frac.frac == i) & (events_frac.geoeff != 1.0))[0]]
    
    X_bzm, Y_tau = np.mgrid[bmin:bmax:db, tmin:tmax:dt]
    positions = np.vstack([X_bzm.ravel(), Y_tau.ravel()]).T
    values = np.vstack([gbzm.values, gtau.values]).T
    
    kernel_bzm_tau_e0 = KernelDensity(kernel='gaussian', bandwidth=ew).fit(values)
    P0 = np.exp(np.reshape(kernel_bzm_tau_e0.score_samples(positions).T, X_bzm.shape))
    
    fig, ax = plt.subplots()
    c = ax.imshow(np.rot90(P0/P0.max()), extent=(bmin,bmax,tmin,tmax), cmap=plt.cm.gist_earth_r)
    ax.plot(gbzm, gtau, 'k.', markersize=4)
    ax.set_xlim([bmin, bmax])
    ax.set_ylim([tmin, tmax])
    ax.set_xlabel('Bzm')
    ax.set_ylabel('Tau')
    fig.colorbar(c)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.random.rand(2, 100) * 4
hist, xedges, yedges = np.histogram2d(gbzm, gtau, normed =1)
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

# Construct arrays with the dimensions for the 16 bars.
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
plt.show()




    hb1, h, b = np.histogram2d(gbzm.values, gtau.values, normed = 1)
    fig, ax = plt.subplots()
    c = plt.imshow(np.rot90(hb1), interpolation='nearest')
    fig.colorbar(c)



    xgrid = np.arange(-75,75,3)
    v = gbzm.values
    k = stats.gaussian_kde(v)
    P = k(xgrid)
    
    k2 = KernelDensity(kernel='gaussian').fit(v.reshape(-1,1))
    P2 = np.exp(k2.score_samples(xgrid.reshape(-1,1)))
    
    plt.plot(xgrid,P)
    plt.plot(xgrid,P2)
    plt.hist(gbzm.values,normed=1)
    
    
    
    
    #kernel_bzm_tau_e = stats.gaussian_kde(values, bw_method = ew)
    #P_bzm_tau_e0= np.reshape(kernel_bzm_tau_e(positions).T, X_bzm.shape)

    fig, ax = plt.subplots()
    c = ax.imshow(np.rot90(P_bzm_tau_e0), extent=(-75,75,0,75), cmap=plt.cm.gist_earth_r)
    ax.plot(gbzm, gtau, 'k.', markersize=4)
    ax.set_xlim([-75, 75])
    ax.set_ylim([0, 75])
    ax.set_xlabel('Bzm')
    ax.set_ylabel('Tau')
    fig.colorbar(c)





def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    #xx, yy = np.mgrid[x.min():x.max():100j, 
    #                  y.min():y.max():100j]
    
    xx, yy = np.mgrid[-75:75:50j, 
                      0:75:50j]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)



m1 = np.random.normal(size=1000)
m2 = np.random.normal(scale=0.5, size=1000)

x, y = m1 + m2, m1 - m2

xx, yy, zz = kde2D(x, y, 1.0)







# Generate random data.
n = 1000
m1, m2 = np.random.normal(-3., 3., size=n), np.random.normal(-3., 3., size=n)
# Define limits.
xmin, xmax = min(m1), max(m1)
ymin, ymax = min(m2), max(m2)
ext_range = [xmin, xmax, ymin, ymax]
# Format data.
x, y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([x.ravel(), y.ravel()])
values = np.vstack([m1, m2])

# Define some point to evaluate the KDEs.
x1, y1 = 0.5, 0.5
# Bandwidth value.
bw = 0.15

# -------------------------------------------------------
# Perform a kernel density estimate on the data using scipy.
# **Bandwidth needs to be scaled to match Sklearn results**
kernel = stats.gaussian_kde(
    values, bw_method=bw/np.asarray(values).std(ddof=1))
# Get KDE value for the point.
iso1 = kernel((x1, y1))
print 'iso1 = ', iso1[0]

# -------------------------------------------------------
# Perform a kernel density estimate on the data using sklearn.
kernel_sk = KernelDensity(kernel='gaussian', bandwidth=bw).fit(zip(*values))
# Get KDE value for the point. Use exponential since sklearn returns the
# log values
iso2 = np.exp(kernel_sk.score_samples([[x1, y1]]))
print 'iso2 = ', iso2[0]


# Plot
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(1, 2)

# Scipy
plt.subplot(gs[0])
plt.title("Scipy", x=0.5, y=0.92, fontsize=10)
# Evaluate kernel in grid positions.
k_pos = kernel(positions)
kde = np.reshape(k_pos.T, x.shape)
plt.imshow(np.rot90(kde), cmap=plt.cm.YlOrBr, extent=ext_range)
plt.contour(x, y, kde, 5, colors='k', linewidths=0.6)

# Sklearn
plt.subplot(gs[1])
plt.title("Sklearn", x=0.5, y=0.92, fontsize=10)
# Evaluate kernel in grid positions.
k_pos2 = np.exp(kernel_sk.score_samples(zip(*positions)))
kde2 = np.reshape(k_pos2.T, x.shape)
plt.imshow(np.rot90(kde2), cmap=plt.cm.YlOrBr, extent=ext_range)
plt.contour(x, y, kde2, 5, colors='k', linewidths=0.6)

fig.tight_layout()
plt.savefig('KDEs', dpi=300, bbox_inches='tight')