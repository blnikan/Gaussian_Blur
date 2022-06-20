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
pix_r = 3
sigma = 1.5
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
    print("pos_x\n",pos_x)
    print('pos_y\n',pos_y)
    res = gauss_func(sigma_,pos_x) * gauss_func(sigma_,pos_y)
    
    return res
    
def Weight_Matrix_1r(sigma_): #create weight matrix of radius = 1
    
    pos_x = np.array([[-1,0,1],
                     [-1,0,1],
                     [-1,0,1]],dtype= np.int32)

    pos_y = np.array([[1,1,1],
                     [0,0,0],
                     [-1,-1,-1]],dtype= np.int32)

    
    res = gauss_func(sigma_,pos_x) * gauss_func(sigma_,pos_y)
    
    return res
    
#------------------------ FUNCTIONS ------------------------
#------------------------ LOAD IMAGE ------------------------
#im = Image.open('/home/nik/Documents/Gaussian_Blur/pic.jpg')
#im = np.array(im)
im = cv.imread('/home/nik/Documents/Gaussian_Blur/pic.jpg')[:,:,::-1]
im_res =  np.zeros(im.shape)
#------------------------ LOAD IMAGE ------------------------
#------------------------  ------------------------
Weight_Matrix = Weight_Matrix_(sigma,pix_r)


for i in range (pix_r,im.shape[0] - pix_r):

    for j in range (pix_r,im.shape[1] - pix_r):
        #0-rojo (red)
        #1-verde (green)
        #2-azul (blue)


        temp_small_im = im[i - pix_r : i + 2 * pix_r, j - pix_r : j + 2 * pix_r , : ]
        
        #red_one =  temp_small_im[:,:,0] * Weight_Matrix  
        #red_one = int(np.sum(temp_small_im[:,:,0] * Weight_Matrix ))
        #green_one = temp_small_im[:,:,1] * Weight_Matrix 
        #green_one = int(np.sum(temp_small_im[:,:,1] * Weight_Matrix ))
        #blue_one =  temp_small_im[:,:,2] * Weight_Matrix 
        #blue_one = int(np.sum(temp_small_im[:,:,2] * Weight_Matrix ))
        
        #if red_one >= 255:
        #    red_one = 254
        #if green_one >= 255:
        #    green_one = 254
        #if blue_one >= 255:
        #    blue_one = 254
        
        
        
        im_res[i,j,0],im_res[i,j,1],im_res[i,j,2] = int(np.sum(temp_small_im[:,:,0] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,1] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,2] * Weight_Matrix ))



#------------------------  ------------------------
#------------------------ OUTPUT ------------------------
#plt.subplot(121),plt.imshow(im)
#plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(im_res)
#plt.xticks([]), plt.yticks([])

#plt.figure(num= None, figsize=(10,10),dpi=80)
#plt.imshow(im.astype(np.uint8))
plt.figure(num= None, figsize=(10,10),dpi=80)
plt.imshow(im_res.astype(np.uint8))
plt.show()

#new_im = Image.fromarray(im_res.astype(np.uint8))
#new_im.save('g.jpg')


#------------------------ OUTPUT ------------------------














