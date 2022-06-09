#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:07:28 2022

@author: nik

"""
from PIL import Image
import numpy as np
import  math
import time

def gauss_func(stdev, pos):
    return (1 / ( (2 * np.pi * stdev ** 2) ** 0.5 ) ) * np.e ** (-((pos[1] ** 2 + pos[0] ** 2)/(2  *stdev ** 2)))

def gauss_func1(stdev, pos):
    return (1 / ( (2 * math.pi * stdev ** 2) ** 0.5 ) ) * math.e ** (-((pos ** 2)/(2  *stdev ** 2)))


im = Image.open('/home/nik/Documents/Gaussian_Blur/pic.jpg')
im = np.array(im)
pix_r = 1
sigma = 1.5

st = time.start_time
pos = np.array([[(-1,1),(0,1),(1,1)],
       [(-1,0),(0,0),(1,0)],
       [(-1,-1),(0,-1),(1,-1)]],dtype= tuple)

G_m = np.zeros((3,3))

for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
       G_m[i,j] = gauss_func(sigma, pos[i,j])
ed = time.end_time
print(ed)
print(G_m)

box_offsets = [] #TODO generates all offset combos


for i in range(-pix_r,pix_r+1): #TODO generation of possible variations 
    for j in range(-pix_r,pix_r+1):
        box_offsets.append((i,j))

#create Weight matrix

coeff = [gauss_func1(sigma, x)*gauss_func1(sigma, y) for x,y in box_offsets]   

print(coeff)     
