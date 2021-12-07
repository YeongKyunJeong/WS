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

# %%


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

# file_1 = Path(filepath/"2014_06_03/target00/FD20140603_172141A.fts")
# file_2 = Path(filepath/"2014_06_03/target00/FD20140603_172141B.fts")
# save_folder = Path(filepath/'Image/data/2014_06_03_target0')
# cad = 20/60
# re = 'umb'

# vmin = -550000
# vmax =  550000

# pos_list = [[115, 116, 117, 118, 
#               115, 116, 117, 118, 
#               116, 117, 118, 119, 120, 
#               115, 116, 117, 118,
#               112, 113, 114, 115]
#             ,[80, 80, 80 ,80,
#               81, 81, 81, 81,
#               83, 83, 83, 83, 83,
#               88, 88, 88, 88,
#               92, 92, 92, 92]]

pos_list = [[120],[83]]

file_1 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813A.fts")
file_2 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813B.fts")
save_folder = Path(filepath/'Image/data/2017_06_14_target4')
cad = 27/60
re = 'qr'

    
vmin = -700000
vmax =  700000

# pos_list = [[170, 180, 200, 170, 200, 155, 185, 150],
#             [105, 105, 105, 110, 115, 140, 145, 150]] 

pos_list = [[155],[140]]
# =============================================================================
# pos_list = [[160, 165, 185, 190,
#               140, 150, 170, 200,
#               165, 170, 175, 180, 185,
#               140, 145, 165, 170, 175, 180, 195, 200,
#               140, 145, 160, 170, 175, 190,
#               145, 155, 170, 180, 195, 200,
#               145, 150, 155, 170, 175, 180, 185, 190,
#               135, 140, 145, 150, 155, 160, 165, 170, 180,
#               145, 150, 165, 170, 175, 180,
#               115, 130, 135, 150, 155, 160, 165, 175,180, 185,
#               140, 150, 160, 175, 180, 185, 195,
#               120, 125, 135, 140, 145, 150, 155, 170,
#               180, 185, 190, 200] ,
#             [90, 90, 90, 90,
#               95, 95, 95, 95,
#               100, 100, 100, 100, 100,
#               105, 105, 105, 105, 105, 105, 105, 105,
#               110, 110, 110, 110, 110, 110,
#               115, 115, 115, 115, 115, 115,
#               120, 120, 120, 120, 120, 120, 120, 120,
#               130, 130, 130, 130, 130, 130, 130, 130, 130,
#               135, 135, 135, 135, 135, 135,
#               140, 140, 140, 140, 140, 140, 140, 140, 140, 140,
#               145, 145, 145, 145, 145, 145, 145,
#               150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150,]]
# =============================================================================


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
colormap = ['purple','blue', 'deepskyblue', 'lime', 'orange', 'red']

for n in range(N):
    # pos=[100,150]#position on the picture
    pos = [pos_list[0][n], pos_list[1][n]]

    vel_list_H = []
    vel_list_Ca = []
    t_list = []
    pic_list_H = []
    pic_list_Ca = []

    div = a.shape[0]//3
    for t in range(a.shape[0]):  # a.shape[0]=time

        #        div = 147 //3 - 1
        #        for t in range(147):#a.shape[0]=time

        i1 = a[t, pos[0], pos[1], 2]
        i2 = c[t, pos[0], pos[1], 2]

        vel_list_H.append(i1)
        vel_list_Ca.append(i2)
        t_list.append((t+1/2)*cad)
        if (t % div) == 0:
            pic_list_H.append(a[t, :, :, 0])
            pic_list_Ca.append(c[t, :, :, 0])
    t_array = np.array(t_list)
    vel_array_H = np.array(vel_list_H)
    vel_array_Ca = np.array(vel_list_Ca)

    res_H = wl.Wavelet(vel_array_H, cad, pad=True)
    wavelet_H = res_H.wavelet
    period_H = res_H.period
    scale_H = res_H.scale
    coi_H = res_H.coi
    power_H = res_H.power

    res_Ca = wl.Wavelet(vel_array_Ca, cad, pad=True)
    wavelet_Ca = res_Ca.wavelet
    period_Ca = res_Ca.period
    scale_Ca = res_Ca.scale
    coi_Ca = res_Ca.coi
    power_Ca = res_Ca.power
    gws_Ca = res_Ca.gws
    # res_Ca.plot()

    coh = wl.WaveCoherency(wavelet_Ca, t_array, scale_Ca,
                           wavelet_H, t_array, scale_H,)

    cross_wave = coh.cross_wavelet
    phase = coh.wave_phase
    coher = coh.wave_coher

    T = coh.time

    

    ts_Ca_3, wl_Ca_3, per_3, pd_3 = wlfiltering(wavelet_Ca, cad, period_Ca, phase,
                                                scale_Ca, [2, 4])
    # ts_Ca_5, wl_Ca_5, per_5, pd_5 = wlfiltering(wavelet_Ca, cad, period_Ca, phase,
    #                                             scale_Ca, [4,8])

    ts_H_3, wl_H_3, per_3, pd_3 = wlfiltering(wavelet_H, cad, period_H, phase,
                                              scale_H, [2, 4])
    # ts_H_5, wl_H_5, per_5, pd_5 = wlfiltering(wavelet_H, cad, period_H, phase,
    #                                           scale_H, [4,8])

    P_Ca_3 = np.abs(wl_Ca_3)**2

    P_H_3 = np.abs(wl_H_3)**2
    # P_H_5 = np.abs(wl_H_5)**2

    ef_H_3, ef_H_list_3 = Eflux_umb(P_H_3, per_3, pd_3, rho_H, pr_H)
    ef_Ca_3, ef_Ca_list_3 = Eflux_umb(P_Ca_3, per_3, pd_3, rho_Ca, pr_Ca)

    # ef_H_5, ef_H_list_5 = Eflux_umb(P_H_5, per_5, pd_5, rho_H, pr_H)
    # ef_Ca_5,ef_Ca_list_5 = Eflux_umb(P_Ca_5, per_5, pd_5, rho_Ca, pr_Ca)

# =============================================================================
#     if np.abs(ef_H_list_tot[10]) > 0:
#
#         mean_ef_H_tot += ef_H_list_tot
#         mean_ef_H_3 += ef_H_list_3
#         mean_ef_H_5 += ef_H_list_5
#
#         mean_ef_Ca_tot += ef_Ca_list_tot
#         mean_ef_Ca_3 += ef_Ca_list_3
#         mean_ef_Ca_5 += ef_Ca_list_5
#         num_o += 1
#
#     else:
#         num_x += 1
#         pos_nan.append(pos)
#
#
#     if n//20 == 0 :
#         print(n)
#
#     if n == N-1:
#
#         print(num_o,num_x)
#
#         mean_ef_H_tot /= num_o
#         mean_ef_H_3 /=  num_o
#         mean_ef_H_5 /=  num_o
#
#         mean_ef_Ca_tot /=  num_o
#         mean_ef_Ca_3 /=  num_o
#         mean_ef_Ca_5 /=  num_o
# =============================================================================

    ip_Ca = ip(t_array,ts_Ca_3, kind='quadratic')
    ip_H = ip(t_array,ts_H_3, kind='quadratic')
    t4ip = np.linspace(t_array[0],t_array[-1],len(t_array)*5)
    tsip_Ca = ip_Ca(t4ip)
    tsip_H = ip_H(t4ip)

    t_Ca_peak = []
    t_H_peak = []
    amp_Ca_peak = []
    amp_H_peak = []

    if re == 'umb' and pos[1] >= 90:
        for t in range(len(t4ip)-2):
            if (t4ip[t+1]< 4) or ( (t4ip[t+1] > 6) and (t4ip[t+1] < 21.5) ) or (t4ip[t+1] > 33.3):
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
        
    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                    
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
        
        
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1])        
 
    elif re =='umb' and pos[1] == 88:
        for t in range(len(t4ip)-2):
            if (t4ip[t+1]< 3.6) or ( (t4ip[t+1] > 6) and (t4ip[t+1] < 23) ) or (t4ip[t+1] > 33):
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1])    
 
    elif re =='umb' and pos[1] == 83:
        for t in range(len(t4ip)-2):
             if ( (t4ip[t+1]< 24) 
                 or ( (t4ip[t+1] > 27) and (t4ip[t+1] < 38) )  
                 or ( (t4ip[t+1] > 41.3) and (t4ip[t+1] < 52) ) 
                 or (t4ip[t+1] > 55) 
                ) :
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1]) 
                    
    elif re =='umb' and pos[1] == 81:
        for t in range(len(t4ip)-2):
             if ( (t4ip[t+1]< 5.11)  
                 or (t4ip[t+1] > 7.11) 
                ) :
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1])
                    
    elif re =='umb' and pos[1] == 80:
        for t in range(len(t4ip)-2):
             if ( (t4ip[t+1]< 5.11)  
                 or (t4ip[t+1] > 7.11) 
                ) :
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1])   
    
    elif re =='qr' and pos == [170, 105]:
        for t in range(len(t4ip)-2):
             if (t4ip[t+1]< 25.2) or ( (t4ip[t+1] > 29.8) and (t4ip[t+1] < 53.7) )  or  (t4ip[t+1] > 56.2)  :
                 
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1])  
                    
    elif re =='qr' and pos == [180, 105]:
        for t in range(len(t4ip)-2):
             if ( ( (t4ip[t+1] > 0.68) and (t4ip[t+1] < 5.2) ) 
                 or  ( (t4ip[t+1] > 8.47) and (t4ip[t+1] < 14.78) )
                 or  ( (t4ip[t+1] > 17.7) and (t4ip[t+1] < 22.2) )
                 or  ( (t4ip[t+1] > 24.9) )
                ): 
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1])  
                    
    elif re =='qr' and pos == [200, 105]:
        for t in range(len(t4ip)-2):
             if ( ( (t4ip[t+1] > 0.68) and (t4ip[t+1] < 31.87) ) 
                 or  ( (t4ip[t+1] > 34.5) )
                ): 
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1]) 
                    
    elif re =='qr' and pos == [170, 110]:
        for t in range(len(t4ip)-2):
             if (  (t4ip[t+1] < 22.6)  
                 or   ( (t4ip[t+1] > 34.6) and (t4ip[t+1] < 58.6))
                ): 
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1]) 

    elif re =='qr' and pos == [200, 115]:
        for t in range(len(t4ip)-2):
             if ( ( (t4ip[t+1] > 0.9) and (t4ip[t+1] < 33.3) ) 
                 or  ( (t4ip[t+1] > 36.1) and (t4ip[t+1] < 44.3) )
                 or  ( (t4ip[t+1] > 45.3) )
                ): 
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1]) 
                    
    elif re =='qr' and pos == [155, 140]:
        for t in range(len(t4ip)-2):
             if ( (t4ip[t+1] < 34)  
                 or  ( (t4ip[t+1] > 35.5) and (t4ip[t+1] < 51.1) )
                 or  ( (t4ip[t+1] > 54.25) )
                ): 
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1])       
 
    elif re =='qr' and pos == [185, 145]:
        for t in range(len(t4ip)-2):
             if ( (t4ip[t+1] < 1.9)
                or  ( (t4ip[t+1] > 2.7) and (t4ip[t+1] < 4.9) )
                or  ( (t4ip[t+1] > 10.02) and (t4ip[t+1] < 26.6) )
                or  ( (t4ip[t+1] > 28.4) )
                ): 
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1])                      
 
    elif re =='qr' and pos == [150, 150]:
        for t in range(len(t4ip)-2):
             if (( (t4ip[t+1] > 1) and (t4ip[t+1] < 52.9) )
                 or  ( (t4ip[t+1] > 54.4) )
                ): 
      
                if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    
                    
                    t_Ca_peak.append(t4ip[t+1])
                    amp_Ca_peak.append(tsip_Ca[t+1])
                
                if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    
                    t_H_peak.append(t4ip[t+1])
                    amp_H_peak.append(tsip_H[t+1])  
    else:
        for t in range(len(t4ip)-2):
            if np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t]) and np.abs(tsip_Ca[t+1]) > np.abs(tsip_Ca[t+2]):
    

                t_Ca_peak.append(t4ip[t+1])
                amp_Ca_peak.append(tsip_Ca[t+1])
                
            if np.abs(tsip_H[t+1]) > np.abs(tsip_H[t]) and np.abs(tsip_H[t+1]) > np.abs(tsip_H[t+2]):
    

                t_H_peak.append(t4ip[t+1])
                amp_H_peak.append(tsip_H[t+1])
                
            
    t_Ca_peak = np.array(t_Ca_peak)
    t_H_peak = np.array(t_H_peak)
    amp_Ca_peak = np.array(amp_Ca_peak)
    amp_H_peak = np.array(amp_H_peak)



    plt.close()
    save_path_img = Path(save_folder/'{}_{}_dt_coh'.format(pos[1], pos[0]))

    fig = plt.figure(
        # figsize=[8, 10]
        figsize=[9, 10]
    )
    gs = gridspec.GridSpec(5, 1, height_ratios=[1, 0.05, 1, 1, 1], width_ratios=[1])

    gs.update(left=0.08, right=0.95, bottom=0.08,
              top=0.93, wspace=0.1, hspace=0.25)

    ax1 = plt.subplot(gs[0,0])
    axcb = plt.subplot(gs[1,0])
    ax2 = plt.subplot(gs[2,0])
    ax3 = plt.subplot(gs[3,0])
    ax4 = plt.subplot(gs[4,0])


    #cmap = plt.get_cmap('rainbow')

    plt1 = ax1.contourf(t_array, scale_H, coher,
                        colors=colormap,
                        levels=[0.5,0.6, 0.7, 0.8, 0.9, 0.95, 1])
    ax1.set_ylim([4, 2])
    ax1.set_xlim([t_array[0], t_array[-1]])
    ax1.set_yscale('log', base=2)
    ax1.set_ylabel('Wavelet Period [min]')
    
    cb = Colorbar(ax=axcb, mappable=plt1,
                  orientation='horizontal',
                  ticklocation='bottom')
    
    cmax = np.max([np.abs(np.min(pd_3)),np.abs(np.max(pd_3))])
    cmin = -cmax    
    
    plt2 = ax2.imshow(pd_3, extent=[t_array[0], t_array[-1], 4, 2], aspect = 6
                      , cmap='PuOr', vmin = cmin, vmax = cmax )
    ax2.set_yticks([4,2])
    ax2.set_ylabel('Wavelet Period [min]')
    
    plt3= ax3.plot(t_Ca_peak, 60*(t_H_peak - t_Ca_peak))
    plt3= ax3.plot(t_array, [60*np.mean(t_H_peak - t_Ca_peak)]*len(t_array),'--', 
                   label = 'Average = {0:03.2f} [s]'.format(60*np.mean(t_H_peak - t_Ca_peak)))
    plt3= ax3.plot(t_array, ori,'--',color='k')
    ax3.set_xlim([t_array[0], t_array[-1]])
    ax3.set_ylabel('Difference Between Peaks [s]')
    ax3.legend()
    
    plt4 = ax4.plot(t4ip, tsip_Ca, label='Wavelet Function of Ca II 8542', color = 'darkblue')
    plt4 = ax4.plot(t4ip, tsip_H, label='Wavelet Function of H-\u03B1', color = 'tab:red')
    ax4.set_xlabel('Time [min]')
    ax4.set_ylabel('Osc Velocity [km/s]')
    ax4.set_xlim([t_array[0], t_array[-1]])
    plt4 = ax4.plot(t_array, ori,'--',color='k')
    # ax4.legend()
    ax4.vlines(t_Ca_peak, ymin = 0, ymax = amp_Ca_peak, linestyle = 'dashed',colors ='tab:blue',alpha = 0.5)
    ax4.vlines(t_H_peak, ymin = 0, ymax = amp_H_peak, linestyle = 'dashed',colors ='tab:orange',alpha = 0.5)
    plt.savefig(save_path_img, dpi=300)

    plt.close()
   
    save_path_img = Path(save_folder/'{}_{}_wlv'.format(pos[1], pos[0]))



    fig = plt.figure(
        figsize=[12, 10]
    )
    gs = gridspec.GridSpec(6, 2, height_ratios=[1, 0.5, 0.5, 0.5, 0.3,0.04], width_ratios=[1, 1])

    gs.update(left=0.05, right=0.95, bottom=0.08,
              top=0.93, wspace=0.1, hspace=0.4)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1:3, 0:2])
    ax4 = plt.subplot(gs[3:5, 0:2])
    cbax = plt.subplot(gs[5, 0:2])

    # ax1.set_xlabel('Time [min]')
    ax1.set_ylabel('Oscillation Velocity [km/s]')
    plt1 = ax1.plot(t_array, vel_array_Ca-np.mean(vel_array_Ca), label='Ca II 8542'
                    ,alpha = 0.6)
    plt1 = ax1.plot(t_array, ts_Ca_3, label='Wavelet Function', color = 'darkblue')
    ax1.set_xlim([t_array[0], t_array[-1]])
    # ax1.set_yticks([-5, 0, 5])    
    ax1.legend()
    
    # ax2.set_xlabel('Time [min]')
    # ax2.set_ylabel('Oscillation Velocity [km/s]')
    plt2 = ax2.plot(t_array, vel_array_H-np.mean(vel_array_H), color = 'tab:orange'
                    ,label='H-\u03B1', alpha = 0.6)
    plt2 = ax2.plot(t_array, ts_H_3, label='Wavelet Function', color = 'tab:red')
    ax2.set_xlim([t_array[0], t_array[-1]])
    ax2.legend()

    ax3.set_xlabel('Time [min]')
    ax3.set_ylabel('Oscillation Velocity [km/s]')

    plt3 = ax3.plot(t_array, [np.mean(amp_H_peak/amp_Ca_peak)]*len(t_array),'--',
                    color = 'm',
                    label = 'Mean of Peak Amplitudes ratios = {0:03.2f}'.format(np.mean(amp_H_peak/amp_Ca_peak)))
     
    plt3 = ax3.plot(t_array, [np.sqrt(np.mean((amp_H_peak)**2))]*len(t_array),'--',
                    color = 'tab:orange', 
                    label = 'RMS of Peak Amplitudes of H-\u03B1 = {0:03.2f} km/s'.format(np.sqrt(np.mean((amp_H_peak)**2))))
    plt3 = ax3.plot(t_array, [np.sqrt(np.mean((amp_Ca_peak)**2))]*len(t_array),'--',
                    color = 'tab:blue',
                    label = 'RMS of Peak Amplitudes of Ca II 8542 = {0:03.2f} km/s'.format(np.sqrt(np.mean((amp_Ca_peak)**2))))
    plt3 = ax3.plot(t_array, ori, '--', color = 'k', label = None, alpha=0.2)
    plt3 = ax3.plot(t4ip, tsip_Ca, label= None
                    # 'Wavelet Function of Ca II 8542'
                    , color = 'darkblue')
    plt3 = ax3.plot(t4ip, tsip_H, label= None
                    # 'Wavelet Function of H-\u03B1'
                    , color = 'tab:red')
    ax3.set_xlim([t_array[0], t_array[-1]])

    
    # ax3.text(1,2.8,'Mean of Peak Amplitide Ratios = {0:03.2f}'.format(np.mean(amp_H_peak/amp_Ca_peak)),fontsize = 15)
    ax3.legend()



    ax4.set_xlabel('Time [min]')
    ax4.set_ylabel('Wavelet Period [km/s]')
    plt4 = ax4.imshow(pd_3, extent=[t_array[0], t_array[-1], 4, 2],
                      aspect = 5, cmap='PuOr', vmin = cmin, vmax = cmax )
    ax4.set_yticks([4,2])
    ax4.set_xlim([t_array[0], t_array[-1]])


    cb_pd = Colorbar(ax=cbax, mappable=plt4,
                    orientation='horizontal', ticklocation='bottom')
    cbax.set_xlabel('[Degree]'
                    #, position = [0,0]
                    )
    plt.savefig(save_path_img, dpi=300)    
    
#%%
    fig = plt.figure(
        figsize=[13, 10]
    )
    gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,1], width_ratios=[1])

    gs.update(left=0.08, right=0.95, bottom=0.08,
              top=0.93, wspace=0.1, hspace=0.25)

    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])

    plt1 = ax1.plot(t4ip, tsip_Ca, label='Wavelet Function of Ca II 8542', color = 'darkblue')
    plt2 = ax2.plot(t4ip, tsip_H, label='Wavelet Function of H-\u03B1', color = 'tab:red')

    plt3 = ax3.plot(t4ip, tsip_Ca, label='Wavelet Function of Ca II 8542', color = 'darkblue')
    plt3 = ax3.plot(t4ip, tsip_H, label='Wavelet Function of H-\u03B1', color = 'tab:red')

    ax3.vlines(t_Ca_peak, ymin = 0, ymax = amp_Ca_peak, linestyle = 'dashed',colors ='tab:blue',alpha = 0.5)
    ax3.vlines(t_H_peak, ymin = 0, ymax = amp_H_peak, linestyle = 'dashed',colors ='tab:orange',alpha = 0.5)
# 
# 
#     
    ax1.vlines(t_Ca_peak, ymin = 0, ymax = amp_Ca_peak, linestyle = 'dashed',colors ='tab:blue',alpha = 0.5)
    ax2.vlines(t_H_peak, ymin = 0, ymax = amp_H_peak, linestyle = 'dashed',colors ='tab:orange',alpha = 0.5)
#     
    
    