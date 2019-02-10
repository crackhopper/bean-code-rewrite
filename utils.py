from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import re

red_c = 200
green_c = 100
blue_c = 39
chs = [red_c, green_c, blue_c]

def readBeanData(fname):
    with open(fname) as f:
        f.readline()
        f.readline()
        dimension_line = f.readline()
        mo = re.search(r'(\d+).+?(\d+).+?(\d+).+', dimension_line)
        dims = [int(i) for i in mo.group(1,2,3)]
        w = dims[0]
        h = dims[1]
        c = dims[2]
        # print(dimension_line)
    dt = pd.read_table(fname,skiprows=5, header=None, sep='\s+')
    arr = dt.as_matrix()
    arr = arr.reshape(h,c,w)
    arr = arr.transpose(0,2,1)
    return arr

def makeRGB(rawArr, scale=True):
    res=[]
    for c in chs:
        if scale:
            scaler = MinMaxScaler()
            transed = scaler.fit_transform(rawArr[:,:,c])
        else:
            transed = rawArr[:,:,c]
        res.append(transed)
    return np.stack(res).transpose(1,2,0)

def mark_foreground(mbd, percent=0.05):
    pixels = np.prod(mbd.shape)
    bound = int(percent*pixels)
    sort = -np.sort(-mbd, axis=None)
    valve = sort[bound]
    result = (mbd>=valve)
    return result

def mark_background(mbd, percent=0.05):
    pixels = np.prod(mbd.shape)
    bound = int(percent*pixels)
    sort = np.sort(mbd, axis=None)
    valve = sort[bound]
    result = (mbd<=valve)
    return result