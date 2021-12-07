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
test = np.linspace(1,500,500)
test2 = np.linspace(-1,498,500)


test3 = np.sin(np.pi*test/5)
test4 = np.sin(np.pi*test2/5)
#%%
fig = plt.figure()
gs = gridspec.GridSpec([2,1])

ax1 = plt.suubplot(gs[0,0])
ax2 = plt.suubplot(gs[1,0])

#%%
res_test= wl.Wavelet(test3,1)
res_test3= wl.Wavelet(test4,1)


wavelet_test = res_test.wavelet
period_test = res_test.period
scale_test = res_test.scale

wavelet_test3 = res_test3.wavelet


coh = wl.WaveCoherency(wavelet_test, test, scale_test,
                       wavelet_test3, test, scale_test,)


cross_wave = coh.cross_wavelet
phase = coh.wave_phase
coher = coh.wave_coher
#%% pd of selected points of umbra

# =============================================================================
file_1 = Path(filepath/"2014_06_03/target00/FD20140603_172141A.fts")
file_2 = Path(filepath/"2014_06_03/target00/FD20140603_172141B.fts")
save_folder = Path(filepath/'Image/data/2014_06_03_target0')

cad = 20/60
pos_list = [[115, 116, 117, 118, 
              115, 116, 117, 118, 
              116, 117, 118, 119, 120, 
              115, 116, 117, 118,
              112, 113, 114, 115]
            ,[80, 80, 80 ,80,
              81, 81, 81, 81,
              83, 83, 83, 83, 83,
              88, 88, 88, 88,
              92, 92, 92, 92]]
# =============================================================================


# =============================================================================
# file_1 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813A.fts")
# file_2 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813B.fts")
# save_folder = Path(filepath/'Image/data/2017_06_14_target4')
# cad = 27/60
# 
# pos_list = [[170, 180, 200, 170, 200, 155, 185, 150],
#             [105, 105, 105, 110, 115, 140, 145, 150]] 
# =============================================================================

# pos_list = [[160, 165, 185, 190,
#              140, 150, 170, 200,
#              165, 170, 175, 180, 185,
#              140, 145, 165, 170, 175, 180, 195, 200,
#              140, 145, 160, 170, 175, 190,
#              145, 155, 170, 180, 195, 200,
#              145, 150, 155, 170, 175, 180, 185, 190,
#              135, 140, 145, 150, 155, 160, 165, 170, 180,
#              145, 150, 165, 170, 175, 180,
#              115, 130, 135, 150, 155, 160, 165, 175,180, 185,
#              140, 150, 160, 175, 180, 185, 195,
#              120, 125, 135, 140, 145, 150, 155, 170,
#              180, 185, 190, 200] ,
#             [90, 90, 90, 90,
#              95, 95, 95, 95,
#              100, 100, 100, 100, 100,
#              105, 105, 105, 105, 105, 105, 105, 105,
#              110, 110, 110, 110, 110, 110,
#              115, 115, 115, 115, 115, 115,
#              120, 120, 120, 120, 120, 120, 120, 120,
#              130, 130, 130, 130, 130, 130, 130, 130, 130,
#              135, 135, 135, 135, 135, 135,
#              140, 140, 140, 140, 140, 140, 140, 140, 140, 140,
#              145, 145, 145, 145, 145, 145, 145,
#              150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150,]]



hdulist = fits.open(file_1)


a = hdulist[0].data
b = hdulist[0].header


hdulist = fits.open(file_2)

c = hdulist[0].data
d = hdulist[0].header

# pos_list = [[115],[80]] 


level = umbpos(a)[3]
N = np.shape(pos_list)[1]

num_o = 0
num_x = 0
pos_nan = []


for n in range(N):
    plt.close()
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


    ts_H_3, wl_H_3, per_3, pd_3 = wlfiltering(wavelet_H, cad, period_H, phase,
                                                  scale_Ca, [2, 4])

    save_path = Path(save_folder/'{}_{}_phase'.format(pos[1], pos[0]))


#        fig = plt.figure(figsize=(20,8))
    fig = plt.figure(figsize=(15, 9))

    gs = gridspec.GridSpec(3, 4, height_ratios=[0.04,1,2.6], width_ratios=[1, 1, 1, 1])

#            3,8, height_ratios=[2.5,1,0.04], width_ratios=[1,1,1,1,1,1,1,1])
    fig.show()
    gs.update(left=0.03, right=0.97, top=0.92,
              bottom=0.02, wspace=0.1, hspace=0.05)


    vmin = np.min(pic_list_H[0])
    vmax = np.max(pic_list_H)

    ax_list = []
    add_cir_list = []
    for i in range(4):
        ax_list.append(plt.subplot(gs[2, i]))
        plt2 = ax_list[i].imshow(pic_list_H[i], origin='lower'  # ,vmin=7000,vmax=9000
                                  #                                          ,vmin=2500,vmax=8500
                                  , vmin=vmin, vmax=vmax
                                   )
        # ax_list[i].contour(a[0,:,:,0], levels=umbpos(a)[3], colors='1', alpha = 0.8)
        add_cir_list.append(plt.Circle(
            (pos[1], pos[0]), 3, lw=2, fill=False, color='red'))
        
    #     add_cir_list.append(plt.Circle((pos[1],pos[0]),7,fill=False,color='red'))
        ax_list[i].add_patch(add_cir_list[i])
        plt2.axes.xaxis.set_ticks([])  # erase tick of axes
        plt2.axes.yaxis.set_ticks([])

    cmax = np.max([np.abs(np.min(pd_3)),np.abs(np.max(pd_3))])
    cmin = -cmax   

    ax3 = plt.subplot(gs[1, 0:4])
    plt3 = ax3.imshow(pd_3, extent=[t_array[0], t_array[-1], 4, 2],
                      aspect =4, cmap='PuOr', vmin = cmin, vmax = cmax )
    ax3.set_yticks([4,2])
    ax3.set_ylabel('Wavelet Period [min]')
    ax3.set_xlabel('Time [min]')
    

    cbax = plt.subplot(gs[0, 0:4])
    cb = Colorbar(ax=cbax, mappable=plt3,
                  orientation='horizontal', ticklocation='top')
    cbax.set_xlabel('[degree]')
    cbax.set_title('Phase difference of ({},{})'.format(pos[1], pos[0]), FONTSIZE = 15)
    plt.savefig(save_path, dpi=300)
    # plt.close()
#%%
# file_1 = Path(filepath/"2014_06_03/target00/FD20140603_172141A.fts")
# file_2 = Path(filepath/"2014_06_03/target00/FD20140603_172141B.fts")
# save_folder = Path(filepath/'Image/data/2014_06_03_target0')

# cad = 20/60
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
# # =============================================================================


file_1 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813A.fts")
file_2 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813B.fts")
save_folder = Path(filepath/'Image/data/2017_06_14_target4')
cad = 27/60

pos_list = [[170, 180, 200, 170, 200, 155, 150],
            [105, 105, 105, 110, 115, 140, 150]] 




hdulist = fits.open(file_1)


a = hdulist[0].data
b = hdulist[0].header


hdulist = fits.open(file_2)

c = hdulist[0].data
d = hdulist[0].header

# pos_list = [[115],[80]] 


level = umbpos(a)[3]
N = np.shape(pos_list)[1]

num_o = 0
num_x = 0
pos_nan = []


# for n in range(1):
for n in range(N):
    plt.close()
    # pos=[100,150]#position on the picture
    pos = [pos_list[0][n], pos_list[1][n]]
    print(pos)

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


    ts_H_3, wl_H_3, per_3, pd_3 = wlfiltering(wavelet_H, cad, period_H, phase,
                                                  scale_Ca, [2, 4])

    save_path = Path(save_folder/'{}_{}_phase_tot_pose'.format(pos[1], pos[0]))


#        fig = plt.figure(figsize=(20,8))
    fig = plt.figure(
        figsize=(15, 9)
                     )

    gs = gridspec.GridSpec(3, 4, height_ratios=[0.04,1,2.6], width_ratios=[1, 1, 1, 1])

#            3,8, height_ratios=[2.5,1,0.04], width_ratios=[1,1,1,1,1,1,1,1])
    fig.show()
    gs.update(left=0.03, right=0.97, top=0.92,
              bottom=0.02, wspace=0.1, hspace=0.05)


    vmin = np.min(pic_list_H[0])
    vmax = np.max(pic_list_H)

    tot_cr = []
    circle_list1 = []
    circle_list2 = []
    circle_list3 = []
    circle_list4 = []
    
    ax_list = []
    add_cir_list = []
    for i in range(4):
        ax_list.append(plt.subplot(gs[2, i]))
        plt2 = ax_list[i].imshow(pic_list_H[i], origin='lower'  # ,vmin=7000,vmax=9000
                                  #                                          ,vmin=2500,vmax=8500
                                  , vmin=vmin, vmax=vmax
                                   )
        # ax_list[i].contour(a[0,:,:,0], levels=umbpos(a)[3], colors='1', alpha = 0.8)
        for n in range(N):
    
            # pos=[100,150]#position on the picture
            poss = [pos_list[0][n], pos_list[1][n]]
        
            circle_list1.append(plt.Circle((poss[1],poss[0]),radius = 3, lw = 2, fill= False,
                                          color = 'red'))
            
            tot_cr.append(circle_list1)
            circle_list2.append(plt.Circle((poss[1],poss[0]),radius = 3, lw = 2, fill= False,
                                          color = 'red'))   

            tot_cr.append(circle_list2)
            circle_list3.append(plt.Circle((poss[1],poss[0]),radius = 3, lw = 2, fill= False,
                                          color = 'red'))

            tot_cr.append(circle_list3)
            circle_list4.append(plt.Circle((poss[1],poss[0]),radius = 3, lw = 2, fill= False,
                                          color = 'red'))

            tot_cr.append(circle_list4)
        for j in range(N): 
            
            #     add_cir_list.append(plt.Circle((pos[1],pos[0]),7,fill=False,color='red'))
            plt.gca().add_artist(tot_cr[i][j])
        
    #     add_cir_list.append(plt.Circle((pos[1],pos[0]),7,fill=False,color='red'))

        plt2.axes.xaxis.set_ticks([])  # erase tick of axes
        plt2.axes.yaxis.set_ticks([])

    cmax = np.max([np.abs(np.min(pd_3)),np.abs(np.max(pd_3))])
    cmin = -cmax   

    ax3 = plt.subplot(gs[1, 0:4])
    plt3 = ax3.imshow(pd_3, extent=[t_array[0], t_array[-1], 4, 2],
                      aspect =4, cmap='PuOr', vmin = cmin, vmax = cmax )
    ax3.set_yticks([4,2])
    ax3.set_ylabel('Wavelet Period [min]')
    ax3.set_xlabel('Time [min]')
    

    cbax = plt.subplot(gs[0, 0:4])
    cb = Colorbar(ax=cbax, mappable=plt3,
                  orientation='horizontal', ticklocation='top')
    cbax.set_xlabel('[degree]')
    cbax.set_title('Phase difference of ({},{})'.format(pos[1], pos[0]), FONTSIZE = 15)
    plt.savefig(save_path, dpi=300)
    # plt.close()
