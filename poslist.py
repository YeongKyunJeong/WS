from astropy.io import fits
# import os.path as path <os.path
from pathlib import Path
from PIL import Image

import numpy as np

from fisspy.analysis import wavelet as wl
#import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib import gridspec, rcParams, rc
from matplotlib.colorbar import Colorbar

from scipy.interpolate import interp1d as ip


def disable_mplkeymaps():
    rc('keymap',
       fullscreen='',
       home='',
       back='',
       forward='',
       pan='',
       zoom='',
       save='',
       quit='',
       grid='',
       yscale='',
       xscale='',
       all_axes=''
       )


FONTSIZE = 12  # Change it on your computer if you wish.
rcParams.update({'font.size': FONTSIZE})
# %%


def wlfiltering(wl, dt, period, pd,
                scale,
                frange, dj=1/10):
    timeseries = []
    srange = np.where((period >= frange[0]) * (period <= frange[-1]))
    ftwl = [[0]*np.shape(wl)[1]]*len(srange[0])
    pd_a = [[0]*np.shape(wl)[1]]*len(srange[0])
    pd_a = np.array(pd_a, dtype=np.float32)
    ftwl = np.array(ftwl, dtype='complex_')
    i, j = 0, 0
    per = []
    for t in range(np.shape(wl)[1]):
        vnew = 0
        for s in srange[0]:
            vnew += (dj*(dt**0.5)/(0.776*np.pi**(-0.25))) * \
                (np.real(wl[s, t])/(period[s]**0.5))
            ftwl[i][j] = wl[s][t]
            pd_a[i][j] = pd[s][t]

            if t == 0:
                per.append(period[s])
#                print(i,j, ftwl[i][j],wl[s][t])
            i += 1

#        print(wl[],ftwl[:,0])
        i = 0
        j += 1
        timeseries.append(vnew)

    return timeseries, ftwl, per, pd_a


#%%


def coicut(P, pd, coi, time):
    find = np.where(coi >= 8)
    cut_P = [[0]*len(find[0])]*np.shape(P)[0]
    cut_pd = [[0]*len(find[0])]*np.shape(P)[0]
    t_list = []
    i, j = 0, 0
    for t in find[0]:
        i = 0
        for per in range(np.shape(P)[0]):
            cut_P[i][j] = P[per][t]
            cut_pd[i][j] = pd[per][t]
            i += 1
        j += 1
        t_list.append(time[t])

    cut_P = np.array(cut_P)
    cut_pd = np.array(cut_pd)
    t_list = np.array(t_list)

    return cut_P, cut_pd, t_list
# %%


def Eflux(Pnew, period, pd, HorCa, M=1.00784, Gam=1.405):
    Fwarray = [[0]*np.shape(Pnew)[1]]*np.shape(Pnew)[0]
    Fwarray = np.array(Fwarray, dtype=np.float32)
    Flist = []
    if HorCa == 'H':
        #        T = 7050
        rho = 1.474e-12
        P = 9.616e-1
#       h = 1.670*1e3
    elif HorCa == 'Ca':
        #        T = 6720
        rho = 4.045e-12
        P = 2.311
#       h = 1.475*1e3
    Cs = np.sqrt(Gam*P/rho)/100
    for j in range(np.shape(Pnew)[1]):
        Fwsum = 0
        for i in range(np.shape(Pnew)[0]):
            w = 1/period[i]/60
            vp = w*200*1000/(pd[i, j]*np.pi/180)

            Fw = (rho*1e6 * (Pnew[i][j]*(1000**2)) * Cs**2 / vp)
#            print(i,j,Fw)
#            print(period[i],i,w)
#            print(Fwsum)
            Fwarray[i][j] = np.real(Fw)

            Fwsum += Fw
        Flist.append(np.real(Fwsum))
    Flist = np.array(Flist)

    return Fwarray, Flist
# %%
# if HorCa == 'H':
#     #        T = 7050
rho = 1.474e-12
P = 9.616e-1
#       h = 1.670*1e3

#        T = 6720
Gam=1.405
# rho = 4.045e-12
# P = 2.311
#       h = 1.475*1e3
Cs = np.sqrt(Gam*P/rho)/100
print(Cs)
#%%
def Eflux_umb(Pnew, period, pd, rho, P, M=1.00784, Gam=1.405):
    Fwarray = [[0]*np.shape(Pnew)[1]]*np.shape(Pnew)[0]
    Fwarray = np.array(Fwarray, dtype=np.float32)
    Flist = []

    Cs = np.sqrt(Gam*P/rho)/100
    for j in range(np.shape(Pnew)[1]):
        Fwsum = 0
        for i in range(np.shape(Pnew)[0]):
            w = 1/period[i]/60
            vp = w*200*1000/(pd[i, j]*np.pi/180)

            Fw = (rho*1e6 * (Pnew[i][j]*(1000**2)) * Cs**2 / vp)
#            print(i,j,Fw)
#            print(period[i],i,w)
#            print(Fwsum)
            Fwarray[i][j] = np.real(Fw)

            Fwsum += Fw
        Flist.append(np.real(Fwsum))
    Flist = np.array(Flist)

    return Fwarray, Flist

# %%


def umbpos(a):
    y = np.where(a[0, :, :, 0] == np.max(a[0, :, :, 0]))[0][0]
    x = np.where(a[0, :, :, 0] == np.max(a[0, :, :, 0]))[1][0]
    level = [0.5*a[0, y, x, 0]]
    umb = np.where(a[0, :, :, 0] < 0.5*a[0, y, x, 0])
    return umb, y, x, level


# %%
#file_path = '/hae/home/ykjeong/Work' <os.path
filepath = Path("/hae/home/ykjeong/Work")

with open(filepath/'modelM.dat', 'r') as f:
    test2 = f.readlines()
    height = []
    density = []
    pressure = []
    temperature = []
    for i in test2:
        # height.append(float(re.split('  | ',i)[2]))
        temperature.append(float(i.split(' ')[1]))
        height.append(float(i.split(' ')[0]))
        density.append(float(i.split(' ')[3]))
        # density.append(float(i.split(' ')[3])*9.109e-28
        #                 +float(i.split(' ')[4])*1.673e-24
        #                 +float(i.split(' ')[5])*1.674e-24)
        pressure.append(float(i.split(' ')[2]))


pr = ip(height, pressure)
rho = ip(height, density)


pr_H = pr(1670)
rho_H = rho(1670)
pr_Ca = pr(1475)
rho_Ca = rho(1475)

#%%

# =============================================================================
# file_1 = Path(filepath/"2014_06_03/target00/FD20140603_172141A.fts")
# file_2 = Path(filepath/"2014_06_03/target00/FD20140603_172141B.fts")
# save_folder = Path(filepath/'Image/data/2014_06_03_target0')
# cad = 20/60
# re = 'umb'
# 
# pos_list = [[118, 116, 112]
#             ,[80, 88, 92]]
# =============================================================================

file_1 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813A.fts")
file_2 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813B.fts")
save_folder = Path(filepath/'Image/data/2017_06_14_target4')
cad = 27/60
re = 'qr'

pos_list = [[170, 180, 200, 170, 200, 155, 150],
            [105, 105, 105, 110, 115, 140, 150]] 


hdulist = fits.open(file_1)
# hdulist[0].data hdulist[0].header


#a = hdulist[0].data
#a = a.T
#b = hdulist[0].header

a = hdulist[0].data
b = hdulist[0].header


hdulist = fits.open(file_2)

c = hdulist[0].data
d = hdulist[0].header




level = umbpos(a)[3]
N = np.shape(pos_list)[1]

num_o = 0
num_x = 0
pos_nan = []
ori = [0]*np.shape(a)[0]

plt.close()
circle_list = []

for n in range(N):
    
    # pos=[100,150]#position on the picture
    pos = [pos_list[0][n], pos_list[1][n]]

    circle_list.append(plt.Circle((pos[1],pos[0]),radius = 2, lw = 2, fill= False,
                                  color = 'red'))
pic_list_H = []
t_list = []
div = a.shape[0]//3
for t in range(a.shape[0]):  # a.shape[0]=time

    #        div = 147 //3 - 1
    #        for t in range(147):#a.shape[0]=time

    t_list.append((t+1/2)*cad)
    if t  == 0:
        pic_list_H.append(a[t, :, :, 0])
            

fig = plt.figure(figsize=(6.3,8))

gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1])

#            3,8, height_ratios=[2.5,1,0.04], width_ratios=[1,1,1,1,1,1,1,1])
fig.show()
gs.update(left=0.03, right=0.97, top=0.95,
          bottom=0.05, wspace=0.1, hspace=0.05)

# =============================================================================
# gs.update(left=0.03, right=0.97, top=0.92,
#           bottom=0.02, wspace=0.1, hspace=0.05)
# =============================================================================


vmin = np.min(pic_list_H[0])
vmax = np.max(pic_list_H)
    
ax_list = []

ax_list.append(plt.subplot(gs[0, 0]))
plt2 = ax_list[0].imshow(pic_list_H[0], origin='lower'  # ,vmin=7000,vmax=9000
                          #                                          ,vmin=2500,vmax=8500
                          , vmin=vmin, vmax=vmax
                           )
if re == 'umb' :
    ax_list[0].contour(a[0,:,:,0], levels=umbpos(a)[3], colors='1', alpha = 0.8)
    ax_list[0].set_title('Umbra : June 3, 2014', FONTSIZE = 15)
else:
    ax_list[0].set_title('QR : June 14, 2017', FONTSIZE = 15)
    
for j in range(N): 
    
    #     add_cir_list.append(plt.Circle((pos[1],pos[0]),7,fill=False,color='red'))
    plt.gca().add_artist(circle_list[j])
# plt2.axes.xaxis.set_ticks([])  # erase tick of axes
# plt2.axes.yaxis.set_ticks([])
  
plt.show()
save_path = Path(save_folder/'position')
plt.savefig(save_path, dpi=300)

