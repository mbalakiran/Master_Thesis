import pandas as pd
import os
import glob
import numpy as np
import itertools as IT
import csv
import csv as cv 
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter
import tkinter.filedialog
from numpy import genfromtxt
from pathlib import Path
import torch
import torchvision
from pylab import *
from itertools import product
from collections import Counter

path = 'C:\\Users\\makn0023\\Desktop\\Thesis'
os.chdir(path)

df2= pd.read_csv("LOD.csv", usecols=[6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36])
# df2=df2.drop(0, axis=0)
# df2=df2.reset_index()
# df2=df2.drop(['index'], axis=1)
df2.to_csv('LOD1.csv', index=False)

from numpy import genfromtxt
my_data = genfromtxt('LOD1.csv', delimiter=',')

my_data = np.delete(my_data, 0,0)

za = []
for i in product([0,1,2,3],repeat=8):
    M = np.zeros((4,4))
    M[i[0]][i[0]] += 1.0
    for j in range(7):
        M[i[j]][i[j+1]] += 1.0
    if np.random.random()<0.01:
        #print(i)
        print(M)
        za.append(M)
        
len(za)
path = 'C:\\Users\\makn0023\\Desktop\\Thesis\\LOD'
os.chdir(path)

abgf = []
for i in range(len(za)):
    abcdf = (my_data[0].flatten()*za[i].flatten()).sum()
    abgf.append(abcdf)
print(abgf)



gf = []
for i in range(1):
    for j in range(len(za)):
        g=((my_data[0].flatten()*za[j].flatten()).sum())
        gf.append(g)
        print(g)
        #gf = pd.DataFrame(data=g)
print(size(gf))
        #np.savetxt(F"lod{i}.csv", gf, delimiter=",")