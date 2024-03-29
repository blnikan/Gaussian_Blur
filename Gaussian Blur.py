#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:07:28 2022

@author: nik

"""
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
#------------------------ CONSTS ------------------------
pix_r = 5
sigma = 3
#------------------------ CONSTS ------------------------
#------------------------  FUNCTIONS  ------------------------

def gauss_func(sigma_, pos): # Gaussian function (sigma_ - sigma ; pos - coordinate  in Gaussian function)
    return (1 / ( (2 * np.pi * sigma_ ** 2) ** 0.5 ) ) * np.e ** (-((pos ** 2)/(2  * sigma_ ** 2)))

def Weight_Matrix_(sigma_, rad_):
    
    pos_x = np.zeros([2*rad_+1,2*rad_+1])
    pos_y = np.zeros([2*rad_+1,2*rad_+1])
    tepm_x = -rad_
    #temp_y = 0
    for i in range(pos_x.shape[0]):
        
        for j in range(pos_x.shape[1]):
            pos_x[j,i] = tepm_x
            pos_y[i,j] = -1*tepm_x
                        
        tepm_x += 1
    #print("pos_x\n",pos_x)
    #print('pos_y\n',pos_y)
    res = gauss_func(sigma_,pos_x) * gauss_func(sigma_,pos_y) 
    
    return res / np.sum(res)
def padding(img,rad_):
    H = img.shape[0]
    W = img.shape[1]
    C = 3
    
    res = np.zeros((H+rad_*2,W+rad_*2,C))
    res[rad_:H+rad_, rad_:W+rad_] = img
    
    #Pad the first/last two col and row
    res[rad_:H+rad_,0:rad_,:C]=img[:,0:1,:C]
    res[H+rad_:H+rad_*2,rad_:W+rad_,:]=img[H-1:H,:,:]
    res[rad_:H+rad_,W+rad_:W+rad_*2,:]=img[:,W-1:W,:]
    res[0:rad_,rad_:W+rad_,:C]=img[0:1,:,:C]
    
    #Pad the missing eight points
    res[0:rad_,0:rad_,:C]=img[0,0,:C]
    res[H+rad_:H+rad_*2,0:rad_,:C]=img[H-1,0,:C]
    res[H+rad_:H+rad_*2,W+rad_:W+rad_*2,:C]=img[H-1,W-1,:C]
    res[0:rad_,W+rad_:W+rad_*2,:C]=img[0,W-1,:C]
    
    
    return res
    
#------------------------ FUNCTIONS ------------------------
#------------------------ LOAD IMAGE ------------------------
im = Image.open('/home/nik/Documents/pic.png')
im = np.array(im)

H = im.shape[0]
W = im.shape[1]

im = padding(im,pix_r)

im_res =  np.zeros(im.shape)

#------------------------ LOAD IMAGE ------------------------

#------------------------------------------------
Weight_Matrix = Weight_Matrix_(sigma,pix_r)
print('sum of Weight_Matrix = ',np.sum(Weight_Matrix))
#a = input('pause')


for i in range (pix_r,im.shape[0] - pix_r):
    print(f'i: {i} from im.shape[0] - pix_r')

    for j in range (pix_r,im.shape[1] - pix_r):
        #0-rojo (red)
        #1-verde (green)
        #2-azul (blue)

        temp_small_im = im[i - pix_r : i + 1 + pix_r, j - pix_r : j + 1 + pix_r , : ]
        im_res[i,j,0],im_res[i,j,1],im_res[i,j,2] = int(np.sum(temp_small_im[:,:,0] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,1] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,2] * Weight_Matrix ))

#------------------------------------------------


#------------------------ OUTPUT ------------------------
new_im = Image.fromarray(im_res[pix_r:H+pix_r, pix_r:W+pix_r].astype(np.uint8))
new_im.save('/home/nik/Documents/Gaussian_Blur/pic.tif')


#------------------------ OUTPUT ------------------------
