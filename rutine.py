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

file_1 = Path(filepath/"2014_06_03/target00/FD20140603_172141A.fts")
file_2 = Path(filepath/"2014_06_03/target00/FD20140603_172141B.fts")
#save_path = Path(filepath/'Image/2017_06_14_target4/PhaseDiff_{}_{}.png'.format(100,150))
cad = 20/60


hdulist = fits.open(file_1)


a = hdulist[0].data
b = hdulist[0].header


hdulist = fits.open(file_2)

c = hdulist[0].data
d = hdulist[0].header


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
level = umbpos(a)[3]
N = np.shape(pos_list)[1]

num_o = 0
num_x = 0
pos_nan = []

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


    ts_H_3, wl_H_3, per_3, pd_3 = wlfiltering(wavelet_H, cad, period_H, phase,
                                                  scale_Ca, [2, 4])

    save_path = Path(filepath/'Image/2014_06_03_target0/pd_search/phase_{}_{}'.format(pos[1], pos[0]))


#        fig = plt.figure(figsize=(20,8))
    fig = plt.figure(figsize=(20, 12))

    gs = gridspec.GridSpec(3, 4, height_ratios=[
                           1.5,0.04,1], width_ratios=[1, 1, 1, 1])

#            3,8, height_ratios=[2.5,1,0.04], width_ratios=[1,1,1,1,1,1,1,1])
    fig.show()
    gs.update(left=0.03, right=0.97, top=0.96,
              bottom=0.04, wspace=0.05, hspace=0.025)


    vmin = np.min(pic_list_H)
    vmax = np.max(pic_list_H)

    ax_list = []
    add_cir_list = []
    for i in range(4):
        ax_list.append(plt.subplot(gs[0, i]))
        plt2 = ax_list[i].imshow(pic_list_H[i], origin='lower'  # ,vmin=7000,vmax=9000
                                 #                                          ,vmin=2500,vmax=8500
                                 , vmin=vmin, vmax=vmax

                                 )
        add_cir_list.append(plt.Circle(
            (pos[1], pos[0]), 3, lw=2, fill=False, color='red'))
    #     add_cir_list.append(plt.Circle((pos[1],pos[0]),7,fill=False,color='red'))
        ax_list[i].add_patch(add_cir_list[i])
        plt2.axes.xaxis.set_ticks([])  # erase tick of axes
        plt2.axes.yaxis.set_ticks([])

    ax3 = plt.subplot(gs[2, 0:4])
    plt3 = ax3.imshow(pd_3, extent=[t_array[0], t_array[-1], 4, 2], aspect =4)
 
    cbax = plt.subplot(gs[1, 0:4])
    cb = Colorbar(ax=cbax, mappable=plt3,
                  orientation='horizontal', ticklocation='top')

    plt.savefig(save_path, dpi=300)
    plt.close()


#%%
fig = plt.figure()
gs = gridspec.GridSpec(3,1)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[2,0])

ax1.plot(test, test3)
ax1.plot(test, test4)

ax2.imshow(np.real(wavelet_test))

ax3.imshow(phase)
#%% all points of umbra pd search
file_1 = Path(filepath/"2014_06_03/target00/FD20140603_172141A.fts")
file_2 = Path(filepath/"2014_06_03/target00/FD20140603_172141B.fts")
#save_path = Path(filepath/'Image/2017_06_14_target4/PhaseDiff_{}_{}.png'.format(100,150))
cad = 20/60


hdulist = fits.open(file_1)


a = hdulist[0].data
b = hdulist[0].header


hdulist = fits.open(file_2)

c = hdulist[0].data
d = hdulist[0].header




pos_list = umbpos(a)[0]
level = umbpos(a)[3]
N = np.shape(pos_list)[1]



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


    ts_H_3, wl_H_3, per_3, pd_3 = wlfiltering(wavelet_H, cad, period_H, phase,
                                                  scale_Ca, [2, 4])

    save_path = Path(filepath/'Image/2014_06_03_target0/pd_search/phase_{}_{}'.format(pos[1], pos[0]))


#        fig = plt.figure(figsize=(20,8))
    fig = plt.figure(figsize=(20, 12))

    gs = gridspec.GridSpec(3, 4, height_ratios=[
                           1.5,0.04,1], width_ratios=[1, 1, 1, 1])

#            3,8, height_ratios=[2.5,1,0.04], width_ratios=[1,1,1,1,1,1,1,1])
    fig.show()
    gs.update(left=0.03, right=0.97, top=0.96,
              bottom=0.04, wspace=0.05, hspace=0.025)


    vmin = np.min(pic_list_H)
    vmax = np.max(pic_list_H)

    ax_list = []
    add_cir_list = []
    for i in range(4):
        ax_list.append(plt.subplot(gs[0, i]))
        plt2 = ax_list[i].imshow(pic_list_H[i], origin='lower'  # ,vmin=7000,vmax=9000
                                 #                                          ,vmin=2500,vmax=8500
                                 , vmin=vmin, vmax=vmax

                                 )
        add_cir_list.append(plt.Circle(
            (pos[1], pos[0]), 3, lw=2, fill=False, color='red'))
    #     add_cir_list.append(plt.Circle((pos[1],pos[0]),7,fill=False,color='red'))
        ax_list[i].add_patch(add_cir_list[i])
        plt2.axes.xaxis.set_ticks([])  # erase tick of axes
        plt2.axes.yaxis.set_ticks([])

    ax3 = plt.subplot(gs[2, 0:4])
    plt3 = ax3.imshow(pd_3, extent=[t_array[0], t_array[-1], 4, 2], aspect =4)

    cbax = plt.subplot(gs[1, 0:4])
    cb = Colorbar(ax=cbax, mappable=plt3,
                  orientation='horizontal', ticklocation='top')

    plt.savefig(save_path, dpi=300)
    plt.close()


#%% means of umbra energy flux of all points


# file_1 = Path(filepath/"2018_06_24/target04/FD20180624_192958A.fts")
# file_2 = Path(filepath/"2018_06_24/target04/FD20180624_192958B.fts")
# #save_path = Path(filepath/'Image/2017_06_14_target4/PhaseDiff_{}_{}.png'.format(100,150))
# cad = 19/60

file_1 = Path(filepath/"2014_06_03/target00/FD20140603_172141A.fts")
file_2 = Path(filepath/"2014_06_03/target00/FD20140603_172141B.fts")
#save_path = Path(filepath/'Image/2017_06_14_target4/PhaseDiff_{}_{}.png'.format(100,150))
cad = 20/60


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


mean_ef_Ca_tot = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_Ca_5 = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_Ca_3 = np.array([0]*np.shape(a)[0], dtype=np.float32)

mean_ef_H_tot = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_H_5 = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_H_3 = np.array([0]*np.shape(a)[0], dtype=np.float32)


pos_list = umbpos(a)[0]
level = umbpos(a)[3]
N = np.shape(pos_list)[1]

num_o = 0
num_x = 0
pos_nan = []

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
    #                                             scale_Ca, [4, 8])
    # ts_Ca_tot, wl_Ca_tot, per_tot, pd_tot = wlfiltering(wavelet_Ca, cad, period_Ca, phase,
    #                                                     scale_Ca, [2, 10])

    ts_H_3, wl_H_3, per_3, pd_3 = wlfiltering(wavelet_H, cad, period_H, phase,
                                              scale_H, [2, 4])
    # ts_H_5, wl_H_5, per_5, pd_5 = wlfiltering(wavelet_H, cad, period_H, phase,
    #                                           scale_H, [4, 8])
    # ts_H_tot, wl_H_tot, per_tot, pd_tot = wlfiltering(wavelet_H, cad, period_H, phase,
    #                                                   scale_H, [2, 10])

    P_Ca_3 = np.abs(wl_Ca_3)**2
    # P_Ca_5 = np.abs(wl_Ca_5)**2
    # P_Ca_tot = np.abs(wl_Ca_tot)**2

    P_H_3 = np.abs(wl_H_3)**2
    # P_H_5 = np.abs(wl_H_5)**2
    # P_H_tot = np.abs(wl_H_tot)**2

    # ef_H_tot, ef_H_list_tot = Eflux_umb(P_H_tot, per_tot, pd_tot, rho_H, pr_H)
    # ef_Ca_tot, ef_Ca_list_tot = Eflux_umb(
    #     P_Ca_tot, per_tot, pd_tot, rho_Ca, pr_Ca)

    ef_H_3, ef_H_list_3 = Eflux_umb(P_H_3, per_3, pd_3, rho_H, pr_H)
    ef_Ca_3, ef_Ca_list_3 = Eflux_umb(P_Ca_3, per_3, pd_3, rho_Ca, pr_Ca)

    # ef_H_5, ef_H_list_5 = Eflux_umb(P_H_5, per_5, pd_5, rho_H, pr_H)
    # ef_Ca_5, ef_Ca_list_5 = Eflux_umb(P_Ca_5, per_5, pd_5, rho_Ca, pr_Ca)

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
#     if n//20 == 0:
#         print(n)
# 
#     if n == N-1:
# 
#         print(num_o, num_x)
# 
#         mean_ef_H_tot /= num_o
#         mean_ef_H_3 /= num_o
#         mean_ef_H_5 /= num_o
# 
#         mean_ef_Ca_tot /= num_o
#         mean_ef_Ca_3 /= num_o
#         mean_ef_Ca_5 /= num_o
# =============================================================================
# %%
fig = plt.figure(
    figsize=[20, 10]
)
gs = gridspec.GridSpec(3, 3                       # ,height_ratios=[]
                        , width_ratios=[1, 1, 1]
                       )
gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.1, hspace=0.5)

ax1 = plt.subplot(gs[0, 0:2])
ax2 = plt.subplot(gs[1, 0:2])
ax3 = plt.subplot(gs[2, 0:2])
ax4 = plt.subplot(gs[0:3, 2])

plt1 = ax1.plot(t_array, mean_ef_H_tot, label='H-\u03B1')
plt1 = ax1.plot(t_array, mean_ef_Ca_tot, label='Ca II 8542')
ax1.set_xlim([t_array[0], t_array[-1]])
ax1.set_title('Total Energy Flux')
ax1.legend()
ax1.set_ylabel('Energy Flux [erg/s/cm^2]')

plt2 = ax2.plot(t_array, mean_ef_H_3, label='H-\u03B1')
plt2 = ax2.plot(t_array, mean_ef_Ca_3, label='Ca II 8542')
ax2.set_xlim([t_array[0], t_array[-1]])
ax2.set_title('3 min Energy Flux')
ax2.legend()

plt3 = ax3.plot(t_array, mean_ef_H_5, label='H-\u03B1')
plt3 = ax3.plot(t_array, mean_ef_Ca_5, label='Ca II 8542')
ax3.set_xlim([t_array[0], t_array[-1]])
ax3.set_title('5 min Energy Flux')
ax3.set_xlabel('Time [min]')
ax3.legend()


plt4 = ax4.contour(a[0, :, :, 0], levels=umbpos(a)[3], colors='r')
plt4 = ax4.imshow(a[0, :, :, 0])
# %%
ax4.tick_params(axis='both', which='both',
                left=False,
                # right= False,
                bottom=False,
                # top = False,
                labelleft=False, labelbottom=False)
plt.plot(t_array, mean_ef_H_tot)
plt.plot(t_array, mean_ef_Ca_tot)
# %% continuous points of umbra ef

# file_1 = Path(filepath/"2018_06_24/target04/FD20180624_192958A.fts")
# file_2 = Path(filepath/"2018_06_24/target04/FD20180624_192958B.fts")
# save_path = Path(filepath/'Image/2017_06_14_target4/PhaseDiff_{}_{}.png'.format(100,150))
# cad = 19/60

file_1 = Path(filepath/"2014_06_03/target00/FD20140603_172141A.fts")
file_2 = Path(filepath/"2014_06_03/target00/FD20140603_172141B.fts")
#save_path = Path(filepath/'Image/2017_06_14_target4/PhaseDiff_{}_{}.png'.format(100,150))
cad = 20/60

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


mean_ef_Ca_tot = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_Ca_5 = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_Ca_3 = np.array([0]*np.shape(a)[0], dtype=np.float32)

mean_ef_H_tot = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_H_5 = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_H_3 = np.array([0]*np.shape(a)[0], dtype=np.float32)


# pos_list = [[124,
#              116, 118, 120, 122, 124, 126,
#              112, 114,
#              116, 114,
#              116, 118], [84,
#                          90, 90, 90, 90, 90, 90,
#                          92, 92,
#                          94, 94,
#                          96, 96]]

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
level = umbpos(a)[3]
N = np.shape(pos_list)[1]

num_o = 0
num_x = 0
pos_nan = []

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
    plt.close()
    save_path_img = Path(
        filepath/'Image/data/2014_06_03_target0/EF_{}_{}'.format(pos[1], pos[0]))

    fig = plt.figure(
        figsize=[20, 7]
    )
    gs = gridspec.GridSpec(3, 2, height_ratios=[
                           0.5, 0.04, 0.8], width_ratios=[1, 1])

    gs.update(left=0.05, right=0.95, bottom=0.08,
              top=0.93, wspace=0.1, hspace=0.2)

    ax1 = plt.subplot(gs[0, 0])
    ax3 = plt.subplot(gs[2, 0])

    ax4 = plt.subplot(gs[0, 1])
    ax6 = plt.subplot(gs[2, 1])

    axcb_1 = plt.subplot(gs[1, 0])
    axcb_2 = plt.subplot(gs[1, 1])

    vmin = np.min([np.min(ef_Ca_3), np.min(ef_H_3)])
    vmax = np.max([np.max(ef_Ca_3), np.max(ef_H_3)])

    plt1 = ax1.imshow(ef_Ca_3, vmax=vmax, vmin=vmin, extent=[
                      t_array[0], t_array[-1], 4, 2], aspect =6)
    ax1.set_title('3 min osc Energy flux of Ca II 8542')
    ax1.set_yticks([4, 2])

    plt4 = ax4.imshow(ef_H_3, vmax=vmax, vmin=vmin, extent=[
                      t_array[0], t_array[-1], 4, 2], aspect = 6)
    ax4.set_title('3 min osc Energy flux of H-\u03B1')
    ax4.set_yticks([4, 2])

    plt3 = ax3.plot(t_array, ef_Ca_list_3, label='Ca II 8542')
    plt3 = ax3.plot(t_array, ef_H_list_3, label='H-\u03B1')
    plt6 = ax6.plot(t_array, vel_array_Ca-np.mean(vel_array_Ca), label='Ca II 8542')
    plt6 = ax6.plot(t_array, vel_array_H-np.mean(vel_array_H), label='H-\u03B1')

    ax3.set_xlim([t_array[0], t_array[-1]])
    ax3.legend()
    ax6.set_xlim([t_array[0], t_array[-1]])
    ax6.legend()

    cb_Ca = Colorbar(ax=axcb_1, mappable=plt1,
                     orientation='horizontal', ticklocation='top')
    cb_H = Colorbar(ax=axcb_2, mappable=plt4,
                    orientation='horizontal', ticklocation='top')

    plt.savefig(save_path_img, dpi=300)
    # plt.close()

    save_path_dat_H = Path(
        filepath/'Image/data/2018_06_24_target4/EF_{}_{}_Ca.dat'.format(pos[1], pos[0]))
    save_path_dat_Ca = Path(
        filepath/'Image/data/2018_06_24_target4/EF_{}_{}_H.dat'.format(pos[1], pos[0]))
    with open(save_path_dat_Ca, 'wb') as save:
        np.savetxt(save, ef_Ca_3,  fmt='%1.4e', delimiter=' ')
    with open(save_path_dat_H, 'wb') as save:
        np.savetxt(save, ef_H_3,  fmt='%1.4e', delimiter=' ')

# %%
file_1 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813A.fts")
file_2 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813B.fts")
cad = 27/60

#file_1 = Path(filepath/"2019_06_19/phys/dc/target01/FD20190619_165043A.fts")
#file_2 = Path(filepath/"2019_06_19/phys/dc/target01/FD20190619_165043B.fts")
#cad = 21/60

pos_list = [[90, 160], [90, 165], [90, 185], [90, 190], 
            [95, 140], [95, 150], [95, 170], [95, 200], 
            [100, 165], [100, 170], [100, 175], [100, 180], [100, 185], 
            [105, 140], [105, 145], [105, 165], [105, 170], [105, 175], [105, 180], [105, 195], 
            [105, 200], 
            [110, 140], [110, 145], [110, 160], [110, 170], [110, 175], [110, 190], 
            [115, 145], [115, 155], [115, 170], [115, 180], [115, 195], [115, 200], 
            [120, 145], [120, 150], [120, 155], [120, 170], [120, 175], [120, 180], [120, 185], 
            [120, 190], 
            [130, 135], [130, 140], [130, 145], [130, 150], [130, 155], [130, 160], [130, 165], 
            [130, 170],
            [130, 180],
            [135, 145], [135, 150], [135, 165], [135, 170], [135, 175], [135, 180],
            [140, 115], [140, 130], [140, 135], [140, 150], [140, 155], [140, 160], [140, 165],
            [140, 175], [140, 180], [140, 185],
            [145, 140], [145, 150], [145, 160], [145, 175], [145, 180], [145, 185], [145, 195],
            [150, 120], [150, 125], [150, 135], [150, 140], [150, 145], [150, 150], [150, 155], 
            [150, 170], [150, 180], [150, 185], [150, 190], [150, 200]
            ]


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


for point in pos_list:
    # pos=[100,150]#position on the picture
    pos = [point[1], point[0]]

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
    # P_Ca_5 = np.abs(wl_Ca_5)**2

    P_H_3 = np.abs(wl_H_3)**2
    # P_H_5 = np.abs(wl_H_5)**2

    ef_H_3, ef_H_list_3 = Eflux(P_H_3, per_3, pd_3, 'H')
    ef_Ca_3, ef_Ca_list_3 = Eflux(P_Ca_3, per_3, pd_3, 'Ca')

    # ef_H_5, ef_H_list_5 = Eflux(P_H_5, per_5, pd_5, 'H')
    # ef_Ca_5,ef_Ca_list_5 = Eflux(P_Ca_5, per_5, pd_5, 'Ca')


# vel osc plot
    save_path_img = Path(
        filepath/'Image/data/2017_06_14_target4/EF_{}_{}'.format(pos[1], pos[0]))

    fig = plt.figure(
        figsize=[20, 7]
    )
    gs = gridspec.GridSpec(3, 2, height_ratios=[
                           0.5, 0.04, 0.8], width_ratios=[1, 1])

    gs.update(left=0.05, right=0.95, bottom=0.08,
              top=0.93, wspace=0.1, hspace=0.2)

    ax1 = plt.subplot(gs[0, 0])
    ax3 = plt.subplot(gs[2, 0])

    ax4 = plt.subplot(gs[0, 1])
    ax6 = plt.subplot(gs[2, 1])

    axcb_1 = plt.subplot(gs[1, 0])
    axcb_2 = plt.subplot(gs[1, 1])

    vmin = np.min([np.min(ef_Ca_3), np.min(ef_H_3)])
    vmax = np.max([np.max(ef_Ca_3), np.max(ef_H_3)])

    plt1 = ax1.imshow(ef_Ca_3, vmax=vmax, vmin=vmin, extent=[
                      t_array[0], t_array[-1], 4, 2], aspect =10)
    ax1.set_title('3 min osc Energy flux of Ca II 8542')
    ax1.set_yticks([4, 2])

    plt4 = ax4.imshow(ef_H_3, vmax=vmax, vmin=vmin, extent=[
                      t_array[0], t_array[-1], 4, 2], aspect = 10)
    ax4.set_title('3 min osc Energy flux of H-\u03B1')
    ax4.set_yticks([4, 2])

    plt3 = ax3.plot(t_array, ef_Ca_list_3, label='Ca II 8542')
    plt3 = ax3.plot(t_array, ef_H_list_3, label='H-\u03B1')
    plt6 = ax6.plot(t_array, vel_array_Ca, label='Ca II 8542')
    plt6 = ax6.plot(t_array, vel_array_H, label='H-\u03B1')

    ax3.set_xlim([t_array[0], t_array[-1]])
    ax3.legend()
    ax6.set_xlim([t_array[0], t_array[-1]])
    ax6.legend()

    cb_Ca = Colorbar(ax=axcb_1, mappable=plt1,
                     orientation='horizontal', ticklocation='top')
    cb_H = Colorbar(ax=axcb_2, mappable=plt4,
                    orientation='horizontal', ticklocation='top')

    plt.savefig(save_path_img, dpi=300)
    plt.close()

    save_path_dat_H = Path(
        filepath/'Image/data/2017_06_14_target4/EF_{}_{}_Ca.dat'.format(pos[1], pos[0]))
    save_path_dat_Ca = Path(
        filepath/'Image/data/2017_06_14_target4/EF_{}_{}_H.dat'.format(pos[1], pos[0]))
    with open(save_path_dat_Ca, 'wb') as save:
        np.savetxt(save, ef_Ca_3,  fmt='%1.4e', delimiter=' ')
    with open(save_path_dat_H, 'wb') as save:
        np.savetxt(save, ef_H_3,  fmt='%1.4e', delimiter=' ')

# %%
file_1 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813A.fts")
file_2 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813B.fts")
cad = 27/60

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


mean_ef_Ca_tot = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_Ca_5 = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_Ca_3 = np.array([0]*np.shape(a)[0], dtype=np.float32)

mean_ef_H_tot = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_H_5 = np.array([0]*np.shape(a)[0], dtype=np.float32)
mean_ef_H_3 = np.array([0]*np.shape(a)[0], dtype=np.float32)

# %%
test = 1
test2 = test+1-1
test += 10
print(test, test2)

# %%

pos_list = umbpos(a)[0]
level = umbpos(a)[3]
N = np.shape(pos_list)[1]

num_o = 0
num_x = 0
pos_nan = []

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
    ts_Ca_5, wl_Ca_5, per_5, pd_5 = wlfiltering(wavelet_Ca, cad, period_Ca, phase,
                                                scale_Ca, [4, 8])
    ts_Ca_tot, wl_Ca_tot, per_tot, pd_tot = wlfiltering(wavelet_Ca, cad, period_Ca, phase,
                                                        scale_Ca, [2, 10])

    ts_H_3, wl_H_3, per_3, pd_3 = wlfiltering(wavelet_H, cad, period_H, phase,
                                              scale_H, [2, 4])
    ts_H_5, wl_H_5, per_5, pd_5 = wlfiltering(wavelet_H, cad, period_H, phase,
                                              scale_H, [4, 8])
    ts_H_tot, wl_H_tot, per_tot, pd_tot = wlfiltering(wavelet_H, cad, period_H, phase,
                                                      scale_H, [2, 10])

    P_Ca_3 = np.abs(wl_Ca_3)**2
    P_Ca_5 = np.abs(wl_Ca_5)**2
    P_Ca_tot = np.abs(wl_Ca_tot)**2

    P_H_3 = np.abs(wl_H_3)**2
    P_H_5 = np.abs(wl_H_5)**2
    P_H_tot = np.abs(wl_H_tot)**2

    ef_H_tot, ef_H_list_tot = Eflux_umb(P_H_tot, per_tot, pd_tot, rho_H, pr_H)
    ef_Ca_tot, ef_Ca_list_tot = Eflux_umb(
        P_Ca_tot, per_tot, pd_tot, rho_Ca, pr_Ca)

    ef_H_3, ef_H_list_3 = Eflux_umb(P_H_3, per_3, pd_3, rho_H, pr_H)
    ef_Ca_3, ef_Ca_list_3 = Eflux_umb(P_Ca_3, per_3, pd_3, rho_Ca, pr_Ca)

    ef_H_5, ef_H_list_5 = Eflux_umb(P_H_5, per_5, pd_5, rho_H, pr_H)
    ef_Ca_5, ef_Ca_list_5 = Eflux_umb(P_Ca_5, per_5, pd_5, rho_Ca, pr_Ca)

    if np.abs(ef_H_list_tot[10]) > 0:

        mean_ef_H_tot += ef_H_list_tot
        mean_ef_H_3 += ef_H_list_3
        mean_ef_H_5 += ef_H_list_5

        mean_ef_Ca_tot += ef_Ca_list_tot
        mean_ef_Ca_3 += ef_Ca_list_3
        mean_ef_Ca_5 += ef_Ca_list_5
        num_o += 1

    else:
        num_x += 1
        pos_nan.append(pos)

    if n//20 == 0:
        print(n)

    if n == N-1:

        print(num_o, num_x)

        mean_ef_H_tot /= num_o
        mean_ef_H_3 /= num_o
        mean_ef_H_5 /= num_o

        mean_ef_Ca_tot /= num_o
        mean_ef_Ca_3 /= num_o
        mean_ef_Ca_5 /= num_o
# %%
# %%

file_1 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813A.fts")
file_2 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813B.fts")
cad = 27/60

#file_1 = Path(filepath/"2019_06_19/phys/dc/target01/FD20190619_165043A.fts")
#file_2 = Path(filepath/"2019_06_19/phys/dc/target01/FD20190619_165043B.fts")
#cad = 21/60

pos_list = [[90, 160], [90, 165], [90, 185], [90, 190], [95, 140], [95, 150], [95, 170], [95, 200], [100, 165], [100, 170], [100, 175], [100, 180], [100, 185], [105, 140], [105, 145], [105, 165], [105, 170], [105, 175], [105, 180], [105, 195], [105, 200], [110, 140], [110, 145], [110, 160], [110, 170], [110, 175], [110, 190], [115, 145], [115, 155], [115, 170], [115, 180], [115, 195], [115, 200], [120, 145], [120, 150], [120, 155], [120, 170], [120, 175], [120, 180], [120, 185], [120, 190], [130, 135], [130, 140], [130, 145], [130, 150], [130, 155], [130, 160], [130, 165], [130, 170],
            [130, 180], [135, 145], [135, 150], [135, 165], [135, 170], [135, 175], [135, 180], [
                140, 115], [140, 130], [140, 135], [140, 150], [140, 155], [140, 160], [140, 165], [140, 175],
            [140, 180], [140, 185], [145, 140], [145, 150], [145, 160], [145, 175], [145, 180], [145, 185], [145, 195], [150, 120], [
                150, 125], [150, 135], [150, 140], [150, 145], [150, 150], [150, 155], [150, 170], [150, 180], [150, 185], [150, 190], [150, 200]
            ]


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


for point in pos_list:
    # pos=[100,150]#position on the picture
    pos = [point[1], point[0]]

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
    ts_Ca_5, wl_Ca_5, per_5, pd_5 = wlfiltering(wavelet_Ca, cad, period_Ca, phase,
                                                scale_Ca, [4, 8])
    ts_Ca_tot, wl_Ca_tot, per_tot, pd_tot = wlfiltering(wavelet_Ca, cad, period_Ca, phase,
                                                        scale_Ca, [2, 10])

    ts_H_3, wl_H_3, per_3, pd_3 = wlfiltering(wavelet_H, cad, period_H, phase,
                                              scale_H, [2, 4])
    ts_H_5, wl_H_5, per_5, pd_5 = wlfiltering(wavelet_H, cad, period_H, phase,
                                              scale_H, [4, 8])
    ts_H_tot, wl_H_tot, per_tot, pd_tot = wlfiltering(wavelet_H, cad, period_H, phase,
                                                      scale_H, [2, 10])

    P_Ca_3 = np.abs(wl_Ca_3)**2
    P_Ca_5 = np.abs(wl_Ca_5)**2
    P_Ca_tot = np.abs(wl_Ca_tot)**2

    P_H_3 = np.abs(wl_H_3)**2
    P_H_5 = np.abs(wl_H_5)**2
    P_H_tot = np.abs(wl_H_tot)**2

    ef_H_tot, ef_H_list_tot = Eflux(P_H_tot, per_tot, pd_tot, 'H')
    ef_Ca_tot, ef_Ca_list_tot = Eflux(P_Ca_tot, per_tot, pd_tot, 'Ca')

    ef_H_3, ef_H_list_3 = Eflux(P_H_3, per_3, pd_3, 'H')
    ef_Ca_3, ef_Ca_list_3 = Eflux(P_Ca_3, per_3, pd_3, 'Ca')

    ef_H_5, ef_H_list_5 = Eflux(P_H_5, per_5, pd_5, 'H')
    ef_Ca_5, ef_Ca_list_5 = Eflux(P_Ca_5, per_5, pd_5, 'Ca')


# vel osc plot
    save_path_img = Path(
        filepath/'Image/data/2017_06_14_target4/EF_{}_{}'.format(pos[1], pos[0]))

    fig = plt.figure(
        figsize=[20, 7]
    )
    gs = gridspec.GridSpec(3, 2, height_ratios=[
                           1, 0.04, 1], width_ratios=[1, 1])

    gs.update(left=0.05, right=0.95, bottom=0.08,
              top=0.93, wspace=0.1, hspace=0.08)

    ax1 = plt.subplot(gs[0, 0])
    ax3 = plt.subplot(gs[2, 0])

    ax4 = plt.subplot(gs[0, 1])
    ax6 = plt.subplot(gs[2, 1])

    axcb_1 = plt.subplot(gs[1, 0])
    axcb_2 = plt.subplot(gs[1, 1])

    vmin = np.min([np.min(ef_Ca_tot), np.min(ef_H_tot)])
    vmax = np.max([np.max(ef_Ca_tot), np.max(ef_H_tot)])

    plt1 = ax1.imshow(ef_Ca_tot, vmax=vmax, vmin=vmin,
                      extent=[t_array[0], t_array[-1], 10, 2])
    ax1.set_title('Energy flux of Ca II 8542')

    plt4 = ax4.imshow(ef_H_tot, vmax=vmax, vmin=vmin, extent=[
                      t_array[0], t_array[-1], 10, 2])
    ax4.set_title('Energy flux of H-\u03B1')

    plt3 = ax3.plot(t_array, ef_Ca_list_tot, label='Ca II 8542')
    plt3 = ax3.plot(t_array, ef_H_list_tot, label='H-\u03B1')
    plt6 = ax6.plot(t_array, vel_array_Ca, label='Ca II 8542')
    plt6 = ax6.plot(t_array, vel_array_H, label='H-\u03B1')

    ax3.set_xlim([t_array[0], t_array[-1]])
    ax3.legend()
    ax6.set_xlim([t_array[0], t_array[-1]])
    ax6.legend()

    cb_Ca = Colorbar(ax=axcb_1, mappable=plt1,
                     orientation='horizontal', ticklocation='top')
    cb_H = Colorbar(ax=axcb_2, mappable=plt4,
                    orientation='horizontal', ticklocation='top')

    plt.savefig(save_path_img, dpi=300)
    plt.close()

    save_path_dat_H = Path(
        filepath/'Image/data/2017_06_14_target4/EF_{}_{}_Ca.dat'.format(pos[1], pos[0]))
    save_path_dat_Ca = Path(
        filepath/'Image/data/2017_06_14_target4/EF_{}_{}_H.dat'.format(pos[1], pos[0]))
    with open(save_path_dat_Ca, 'wb') as save:
        np.savetxt(save, ef_Ca_tot,  fmt='%1.4e', delimiter=' ')
    with open(save_path_dat_H, 'wb') as save:
        np.savetxt(save, ef_H_tot,  fmt='%1.4e', delimiter=' ')


# %%


#file_path = '/hae/home/ykjeong/Work' <os.path
# =============================================================================
# filepath = Path("/hae/home/ykjeong/Work")
#
# file_1 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813A.fts")
# file_2 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813B.fts")
# cad = 27/60
# JJ_list = [ 50, 55, 60, 65, 70, 75, 80 ,85, 90, 95,
#            100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
# II_list = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145,
#            150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
# =============================================================================
# =============================================================================
# filepath = Path("/hae/home/ykjeong/Work"
#
# file_1 = Path(filepath/"2019_06_19/phys/dc/target01/FD20190619_165043A.fts")
# file_2 = Path(filepath/"2019_06_19/phys/dc/target01/FD20190619_165043B.fts")
# cad = 21/60
# =============================================================================

# =============================================================================
filepath = Path("/hae/home/ykjeong/Work")
file_1 = Path(filepath/"2018_06_24/target04/FD20180624_192958A.fts")
file_2 = Path(filepath/"2018_06_24/target04/FD20180624_192958B.fts")
# save_path = Path(filepath/'Image/2018_06_24_target4/pd_search/phase_{}_{}'.format(JJ,II))
cad = 19/60
JJ_list = [
    #50, 52, 54, 56, 58, 60, 62, 64,
    66, 68,
    70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
    90, 92, 94, 96, 98
    # , 100
]
II_list = [
    #90, 92, 94, 96,
    98, 100, 102, 104, 106, 108,
    110, 112, 114, 116, 118, 120, 122, 124, 126, 128,
    130, 132, 134, 136, 138, 140, 142
    # , 144, 146
]
# JJ_list = [
#     #50, 52, 54, 56, 58, 60, 62, 64,
#             66, 68,
#             70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
#             90, 92, 94, 96, 98
#           #, 100
#             ]
# II_list = [
#           #90, 92, 94, 96,
#             98, 100, 102, 104, 106, 108,
#             110, 112, 114, 116, 118, 120, 122, 124, 126, 128,
#             130, 132, 134, 136, 138, 140, 142
#           #, 144, 146
#             ]

# =============================================================================


# =============================================================================
# filepath = Path("/hae/home/ykjeong/Work")
# file_1 = Path(filepath/"2018_06_24/target08/FD20180624_205133A.fts")
# file_2 = Path(filepath/"2018_06_24/target08/FD20180624_205133B.fts")
# #save_path = Path(filepath/'Image/2018_06_24_target8/pd_search/phase_{}_{}'.format(JJ,II))
#
# cad = 19/60
# JJ_list = [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
#            70, 72, 74, 76, 78, 80, 82, 84, 86
#             ]
# II_list = [ 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137,
#            139, 141, 143, 145, 147, 149, 151, 153
#             ]
# =============================================================================
# =============================================================================
# filepath = Path("/hae/home/ykjeong/Work")
# file_1 = Path(filepath/"2018_06_24/target10/FD20180624_215229A.fts")
# file_2 = Path(filepath/"2018_06_24/target10/FD20180624_215229B.fts")
# #save_path = Path(filepath/'Image/2018_06_24_target8/pd_search/phase_{}_{}'.format(JJ,II))
#
# cad = 19/60
# JJ_list = [50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
#             70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92
#             ]
# II_list = [ 121, 123, 125, 127, 129, 131, 133, 135, 137,
#             139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159
#             ]
#
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


for II in II_list:
    for JJ in JJ_list:
        # pos=[100,150]#position on the picture
        pos = [II, JJ]

        vel_list_H = []
        vel_list_Ca = []
        t_list = []
        pic_list_H = []
        pic_list_Ca = []

        div = a.shape[0]//3
#        div = a.shape[0]//7
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

        ts_Ca_3, wl_Ca_3, per_3, pd_3 = wlfiltering(
            wavelet_Ca, cad, period_Ca, phase, scale_Ca, [2, 4])
        # ts_Ca_5, wl_Ca_5, per_5, pd_5 = wlfiltering(wavelet_Ca, cad, period_Ca, phase, scale_Ca, [4,8])

        ts_H_3, wl_H_3, per_3, pd_3 = wlfiltering(
            wavelet_H, cad, period_H, phase, scale_H, [2, 4])
        # ts_H_5, wl_H_5, per_5, pd_5 = wlfiltering(wavelet_H, cad, period_H, phase, scale_H, [4,8])
        # ts_Ca, wl_Ca, per, pd = wlfiltering(wavelet_Ca, cad, period_Ca, phase, scale_Ca, [2,8])


# vel osc plot
#        save_path = Path(filepath/'Image/2017_06_14_target4/pd_search/phase_{}_{}'.format(JJ,II))
        save_path = Path(
            filepath/'Image/2018_06_24_target4/pd_search/phase_{}_{}'.format(JJ, II))
#        save_path = Path(filepath/'Image/2018_06_24_target8/pd_search/phase_{}_{}'.format(JJ,II))
        # save_path = Path(filepath/'Image/2018_06_24_target10/pd_search/phase_{}_{}'.format(JJ,II))


#        fig = plt.figure(figsize=(20,8))
        fig = plt.figure(figsize=(20, 12))

        gs = gridspec.GridSpec(3, 4, height_ratios=[
                               1.5, 1, 0.04], width_ratios=[1, 1, 1, 1])

#            3,8, height_ratios=[2.5,1,0.04], width_ratios=[1,1,1,1,1,1,1,1])
        fig.show()
        gs.update(left=0.03, right=0.97, top=0.96,
                  bottom=0.04, wspace=0.05, hspace=0.025)

        ax_list = []
        add_cir_list = []
        for i in range(4):
            ax_list.append(plt.subplot(gs[0, i]))
            plt2 = ax_list[i].imshow(pic_list_H[i], origin='lower'  # ,vmin=7000,vmax=9000
                                     #                                          ,vmin=2500,vmax=8500
                                     , vmin=3100, vmax=8100

                                     )
            add_cir_list.append(plt.Circle(
                (pos[1], pos[0]), 3, lw=2, fill=False, color='red'))
        #     add_cir_list.append(plt.Circle((pos[1],pos[0]),7,fill=False,color='red'))
            ax_list[i].add_patch(add_cir_list[i])
            plt2.axes.xaxis.set_ticks([])  # erase tick of axes
            plt2.axes.yaxis.set_ticks([])

        ax3 = plt.subplot(gs[1, 0:4])
        plt3 = ax3.imshow(pd_3, extent=[t_array[0], t_array[-1], 10, 2])

        cbax = plt.subplot(gs[2, 0:4])
        cb = Colorbar(ax=cbax, mappable=plt2,
                      orientation='horizontal', ticklocation='top')

        plt.savefig(save_path, dpi=300)
        plt.close()

# =============================================================================
#         fig = plt.figure(figsize=(20,15))
#         gs = gridspec.GridSpec(3,4, height_ratios=[2.5,1,0.04], width_ratios=[1,1,1,1])
#         fig.show()
#         gs.update(left=0.03, right=0.97, top = 0.96, bottom = 0.04, wspace=0.05, hspace=0.025)
#
#         ax_list=[]
#         add_cir_list = []
#         for i in range(4):
#                 ax_list.append(plt.subplot(gs[0,i]))
#                 plt2 = ax_list[i].imshow(pic_list_H[i],origin='lower'
# #                                         ,vmin=7000,vmax=9000
#                                           ,vmin=2500,vmax=8500
#                                          )
#                 add_cir_list.append(plt.Circle((pos[1],pos[0]),3,lw = 2,fill=False,color='red'))
#         #     add_cir_list.append(plt.Circle((pos[1],pos[0]),7,fill=False,color='red'))
#                 ax_list[i].add_patch(add_cir_list[i])
#                 plt2.axes.xaxis.set_ticks([]) #erase tick of axes
#                 plt2.axes.yaxis.set_ticks([])
#
#         ax3 = plt.subplot(gs[1,0:4])
#         plt3 = ax3.imshow(pd,extent=[t_array[0],t_array[-1],10,2])
#
#         cbax = plt.subplot(gs[2,0:4])
#         cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal',ticklocation = 'top')
#
#
#         plt.savefig(save_path, dpi = 300)
#         plt.close()
# =============================================================================


# %%


#        save_path = Path(filepath/'Image/2019_06_19_target1/velocity_{}_{}'.format(JJ,II))
#        save_path = Path(filepath/'Image/2017_06_14_target4/velocity_{}_{}'.format(JJ,II))
        save_path = Path(
            filepath/'Image/2017_06_14_target4/pd_search/phase_{}_{}'.format(JJ, II))

        fig = plt.figure()

        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 1)
        # gs.update=()

        ax = plt.subplot(gs[0, 0])
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Velocity [km/s]')
        ax.set_xlim(t_array[0], t_array[-1])

        plt1 = ax.plot(t_array, vel_array_Ca, 'r', label='H-\u03B1')
        plt2 = ax.plot(t_array, vel_array_H, 'k', label='Ca II 8542')
        ax.legend()
        plt.savefig(save_path, dpi=300)

        plt.close()
# %%

        # wavelet

        res_H = wl.Wavelet(vel_array_H, cad, pad=True)
        wavelet_H = res_H.wavelet
        period_H = res_H.period
        scale_H = res_H.scale
        coi_H = res_H.coi
        power_H = res_H.power
        gws_H = res_H.gws
        # res_H.plot()

        res_Ca = wl.Wavelet(vel_array_Ca, cad, pad=True)
        wavelet_Ca = res_Ca.wavelet
        period_Ca = res_Ca.period
        scale_Ca = res_Ca.scale
        coi_Ca = res_Ca.coi
        power_Ca = res_Ca.power
        gws_Ca = res_Ca.gws
        # res_Ca.plot()

        coh = wl.WaveCoherency(wavelet_Ca, t_array, scale_Ca,
                               wavelet_H, t_array, scale_H)

        cross_wave = coh.cross_wavelet
        phase = coh.wave_phase
        coher = coh.wave_coher
        gCoher = coh.global_coher
        gCross = coh.global_cross
        gPhase = coh.global_phase
        power1 = coh.power1
        power2 = coh.power2
        time_out = coh.time
        scale_out = coh.scale
        T = coh.time

        coi = np.array([1]*np.shape(coi_H)[0])

        for i in range(np.shape(coi)[0]):

            if coi_H[i] == 0:
                coi[i] = -20
            else:
                coi[i] = 10*np.log2(coi_H[i]/0.9)

        num = np.linspace(0, np.shape(coi)[0]-1, np.shape(coi)[0])

        # Coherency plot
        save_path = Path(
            filepath/'Image/2019_06_19_target1/PhaseDiff_{}_{}'.format(JJ, II))

        fig = plt.figure(figsize=((t_array[-1]-t_array[0])/4, 32/4 + 1/4 + 2/4)
                         )
        gs = gridspec.GridSpec(2, 1, height_ratios=([1, 32]))
        gs.update(left=0.04, right=0.96, top=0.96, bottom=0.04,
                  hspace=2*2/33)
        # fig.show

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])

        ax2.set_ylabel('Period [min]')
        ax2.set_xlabel('Time [min]')

        colormap = ['blue', 'deepskyblue', 'lime', 'orange', 'red']

        #cmap = plt.get_cmap('rainbow')

        plt1 = ax2.contourf(t_array, scale_H, coher,
                            colors=colormap,
                            levels=[0.6, 0.7, 0.8, 0.9, 0.95, 1])

        # plt1 = ax2.contourf(coher,
        #                    colors=colormap,
        #                    levels = [0.6,0.7,0.8,0.9,0.95,1]
        #                    )

        plt.gca().invert_yaxis()

        # ax2.set_ylim([16,0.9])

        cb = Colorbar(ax=ax1, mappable=plt1,
                      orientation='horizontal',
                      ticklocation='top')
        #plt.clabel(plt1,inline = True, fontsize=10)

        # plt.plot(t_array,10*np.log2(coi_H/0.9),'k--')
        plt.plot(t_array, coi_H, 'k--')

        # Coi
        t = [-1]
        y_fill = [0]
        for i in range(len(t_array)):
            t.append(t_array[i])
            if coi_H[i] == 0:
                y_fill.append(0)
            else:
                y_fill.append(coi_H[i])
        t.append(t_array[-1]+2)
        y_fill.append(0)
        t = np.array(t)
        y_fill = np.array(y_fill)

        plt.fill_between(t, y_fill, 64, hatch='x',
                         alpha=0
                         )

        find = np.where(coher > 0.8, 1, 0)
        coord = np.where(coher > 0.8)
        phase_diff = phase*find

        ax2.set_ylim([16, 1])
        ax2.set_xlim([t_array[0], t_array[-1]])
        ax2.set_yscale('log', base=2)
        # ax2.set_yticks([1,2,4,8,16])

        #plt.contourf(t_array, scale_H,phase)
        # Be ware not to confuse the x,y and row,col

        for i in range(len(coord[0])):
            if scale_H[coord[0][i]] > 2 or coord[0][i] % 2 == 0:

                if coord[0][i] % 2 == 0 and i % 4 == 0:
                    col = coord[1][i]
                    row = coord[0][i]
                    x = t_array[col]
                    y = scale_H[row]
                    dx = 1.5*np.cos(phase_diff[row, col]*np.pi/180)
                    dy = 1.5*np.sin(phase_diff[row, col]*np.pi/180)
                    ax2.annotate('', xy=(x+dx/2, y*2**(-dy/16)),
                                 xytext=(x-dx/2, y*2**(dy/16)),
                                 arrowprops=dict(arrowstyle='->', lw=2, facecolor='black'))

        plt.savefig(save_path, dpi=300)
        plt.close()

##########################################


for II in II_list:
    for JJ in JJ_list:
        # pos=[100,150]#position on the picture
        pos = [II, JJ]

        phase_im_path = Path(
            filepath/'Image/2019_06_19_target1/PhaseDiff_{}_{}.png'.format(JJ, II))
        save_path = Path(
            filepath/'Image/2019_06_19_target1/PhaseDiff_{}_{}_im.png'.format(JJ, II))

        fig = plt.figure(figsize=(8, 11))
        gs = gridspec.GridSpec(3, 2, height_ratios=[
                               0.04, 1, 1], width_ratios=[1, 1])
        fig.show()
        gs.update(left=0.03, right=0.97, top=0.96,
                  bottom=0.04, wspace=0.05, hspace=0.025)

        ax_list = []
        add_cir_list = []
        for i in range(4):
            ax_list.append(plt.subplot(gs[i//2+1, i % 2]))
            plt2 = ax_list[i].imshow(pic_list_H[i], origin='lower'  # ,vmin=7000,vmax=9000
                                     , vmin=6000, vmax=7700
                                     )
            add_cir_list.append(plt.Circle(
                (pos[1], pos[0]), 9, lw=3, fill=False, color='red'))
        #     add_cir_list.append(plt.Circle((pos[1],pos[0]),7,fill=False,color='red'))
            ax_list[i].add_patch(add_cir_list[i])
            plt2.axes.xaxis.set_ticks([])  # erase tick of axes
            plt2.axes.yaxis.set_ticks([])

        cbax = plt.subplot(gs[0, 0:2])
        cb = Colorbar(ax=cbax, mappable=plt2,
                      orientation='horizontal', ticklocation='top')

        plt.savefig(save_path, dpi=300)
        plt.close()


############################################################


# %%
        #file_1 = Path(filepath/"2019_06_19/phys/dc/target01/FD20190619_165043A.fts")
        #file_2 = Path(filepath/"2019_06_19/phys/dc/target01/FD20190619_165043B.fts")


for II in II_list:
    for JJ in JJ_list:

        # pos=[100,150]#position on the picture
        pos = [II, JJ]

        file_1 = Path(
            filepath/'Image/2019_06_19_target1/PhaseDiff_{}_{}.png'.format(JJ, II))
        file_2 = Path(
            filepath/'Image/2019_06_19_target1/PhaseDiff_{}_{}_im.png'.format(JJ, II))

        save_path = Path(
            filepath/'Image/2019_06_19_target1/results/results_{}_{}_im.png'.format(JJ, II))

        im1 = Image.open(file_1)
        im2 = Image.open(file_2)

        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1.2])
        fig.show()
        gs.update(
            left=0.0, right=1,
            wspace=0)

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax1.axis('off')
        ax2.axis('off')
        plt1 = ax1.imshow(im1)
        plt2 = ax2.imshow(im2)

#        plt2.axes.xaxis.set_ticks([]) #erase tick of axes
#        plt2.axes.yaxis.set_ticks([])

        plt.savefig(save_path, dpi=300)
        plt.close()
