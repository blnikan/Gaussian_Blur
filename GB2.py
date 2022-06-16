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

pix_r = 1
sigma = 1.5

#------------------------ CONSTS ------------------------
"""


I = cv2.imread('sarajevo.jpg')[:, :, ::-1]
plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(I)
plt.show()
"""

#vvvvvvvvvvvvvvvvvvvvvvv  FUNCTIONS  vvvvvvvvvvvvvvvvvvvvvvv

def gauss_func(sigma_, pos): # Gaussian function (sigma_ - sigma ; pos - coordinate  in Gaussian function)
    return (1 / ( (2 * np.pi * sigma_ ** 2) ** 0.5 ) ) * np.e ** (-((pos ** 2)/(2  * sigma_ ** 2)))

def Weight_Matrix(sigma_): #create weight matrix of radius = 1
    
    pos_x = np.array([[-1,0,1],
                     [-1,0,1],
                     [-1,0,1]],dtype= np.int32)

    pos_y = np.array([[1,1,1],
                     [0,0,0],
                     [-1,-1,-1]],dtype= np.int32)


    res = gauss_func(sigma_,pos_x) * gauss_func(sigma_,pos_y)
    
    return res
    
#^^^^^^^^^^^^^^^^^^^^^^^ FUNCTIONS ^^^^^^^^^^^^^^^^^^^^^^^^


#------------------------ LOAD IMAGE ------------------------
#im = Image.open('/home/nik/Documents/Gaussian_Blur/pic.jpg')
#im = np.array(im)
im = cv.imread('/home/nik/Documents/Gaussian_Blur/pic.jpg')[:,:,::-1]
im_res =  np.zeros(im.shape)

#------------------------ LOAD IMAGE ------------------------



Weight_Matrix = Weight_Matrix(sigma)

t = im[:,:,0]


for i in range (1,im.shape[0] - 1):

    for j in range (1,im.shape[1] - 1):

        temp_small_im = im[i - pix_r : i + 2 * pix_r, j - pix_r : j + 2 * pix_r , : ]
        
        print(temp_small_im[:,:,0],'\n\n]',Weight_Matrix,'\n\n\n')
        red_one =  temp_small_im[:,:,0]*Weight_Matrix
        print('\n',red_one)
        
        red_one = int(np.sum(red_one))
        print(np.sum(red_one,dtype=np.int32))
        print('\n',red_one)
        
        if red_one >255:
            red_one =255
        
        green_one = temp_small_im[:,:,1]*Weight_Matrix 
        green_one = int(np.sum(green_one))
        
        if green_one >255:
            green_one =255
        
        blue_one =  temp_small_im[:,:,2]*Weight_Matrix 
        blue_one = int(np.sum(blue_one))
        
        
        
        if blue_one >255:
            blue_one =255
        break
        
        im_res[i,j,0] = red_one#int(np.sum(red_one))
        im_res[i,j,1] =  blue_one #int(np.sum(green_one))
        im_res[i,j,2] = green_one#int(np.sum(blue_one))
        
    break
    


#------------------------ OUTPUT ------------------------

#plt.subplot(121),plt.imshow(im)
#plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(im_res)
#plt.xticks([]), plt.yticks([])
plt.figure(num= None, figsize=(10,10),dpi=80)
plt.imshow(im_res)
plt.show()
