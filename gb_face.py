#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:08:45 2022

@author: nik
@author git: https://github.com/blnikan

"""

import numpy as np
import cv2 
import dlib
import face_recognition


def gauss_func(sigma_, pos): 
    return (1 / ( (2 * np.pi * sigma_ ** 2) ** 0.5 ) ) * np.e ** (-((pos ** 2)/(2  * sigma_ ** 2)))

def Weight_Matrix_(sigma_, rad_):
    
    pos_x = np.zeros([2*rad_+1,2*rad_+1])
    pos_y = np.zeros([2*rad_+1,2*rad_+1])
    tepm_x = -rad_

    for i in range(pos_x.shape[0]):
        
        for j in range(pos_x.shape[1]):
            pos_x[j,i] = tepm_x
            pos_y[i,j] = -1*tepm_x
                        
        tepm_x += 1

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


def GauB_1pix(img,i_,j_,rad,sig):
    
    temp_small_im = img[i_ - rad : i_ + 1 + rad, j_ - rad : j_ + 1 + rad , : ]
    Weight_Matrix = Weight_Matrix_(sig,rad)

    return  int(np.sum(temp_small_im[:,:,0] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,1] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,2] * Weight_Matrix ))
    

    

def GauB(img,rad,sig):
    im_res =  np.zeros(img.shape)
    Weight_Matrix = Weight_Matrix_(sig,rad)
    print('sum of Weight_Matrix = ',np.sum(Weight_Matrix))

    for i in range (rad,img.shape[0] - rad):
        print(f'i: {i} from im.shape[0] - rad')
        for j in range (rad,img.shape[1] - rad):

            temp_small_im = img[i - rad : i + 1 + rad, j - rad : j + 1 + rad , : ]

            im_res[i,j,0],im_res[i,j,1],im_res[i,j,2] = int(np.sum(temp_small_im[:,:,0] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,1] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,2] * Weight_Matrix ))

    return im_res


def dec(i0,j0,i1,j1):
    return ( (i0-i1)**2 + (j0-j1)**2 )**0.5
