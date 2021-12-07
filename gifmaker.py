from astropy.io import fits
#import os.path as path <os.path
from pathlib import Path
from PIL import Image
import numpy as np

from fisspy.analysis import wavelet as wl
#import matplotlib as mp
from matplotlib import pyplot as plt
from matplotlib import gridspec, rcParams, rc
from matplotlib.colorbar import Colorbar

import re
from scipy.interpolate import interp1d as ip
from PIL import Image
import imageio
#%%
#file_path = '/hae/home/ykjeong/Work' <os.path
filepath = Path("/hae/home/ykjeong/Work")

# file_1 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813A.fts")
# file_2 = Path(filepath/"2017_06_14/phys/qr/target04/FD20170614_175813B.fts")
# save_folder = Path(filepath/'Image/2017_06_14_6559cont.gif')
# cad = 27/60
# vmin = 7780

file_1 = Path(filepath/"2014_06_03/target00/FD20140603_172141A.fts")
file_2 = Path(filepath/"2014_06_03/target00/FD20140603_172141B.fts")
save_folder = Path(filepath/'Image/2014_06_06_6559cont.gif')
cad = 20/60
vmin = 2900

hdulist = fits.open(file_1)
#hdulist[0].data hdulist[0].header

a = hdulist[0].data
b = hdulist[0].header


hdulist = fits.open(file_2)

c = hdulist[0].data
d = hdulist[0].header

#%%



cont=[]



vmax = np.max(a[:,:,:,0])
k = 255/(vmax-vmin)
div = a.shape[0]//3
#div = 147 //3]
for t in range(a.shape[0]):#a.shape[0]=time
    
    # p2 = np.array([[0]*a.shape[2]]*a.shape[1])
    p = (np.array(a[t,:,:,0])-vmin)*k
    p[p<0] = 0
    p = np.uint8(p)


    # cont.append(k2)
    cont.append(Image.fromarray(p).convert('P'))
#%%
cont[0].save(fp = save_folder, format = 'GIF',save_all = True,
           append_images =cont[1:], optimize = False,duration = 30, loop = 0)
        