#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:27:11 2022

@author: nik
@author git: https://github.com/blnikan

"""
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

#------------------------  FUNCTIONS  ------------------------

def gauss_func(sigma_, pos): # Gaussian function (sigma_ - sigma ; pos - coordinate  in Gaussian function)
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
    




def padding1 ( img , rad_ ):
    
    size_i_new = img.shape[0] + rad_*2
    size_j_new = img.shape[1] + rad_*2
    
    
    res_img = np.zeros([size_i_new,size_j_new,3])
    
    
    res_img[rad_:-rad_,:rad_] = img[:,0,:] # левый край
    #res_img[:rad_,0] = img[0,0]
        
    #res_img[rad_:-rad_,rad_:-rad_,:] = img
    
    return res_img


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

    
#------------------------ FUNCTIONS ------------------------
#------------------------ LOAD IMAGE ------------------------
pix_r = 3
pix_r_MAX = 20
sigma = 7

im = Image.open('/home/nik/Documents/Gaussian_Blur/pic.png')  # /home/nik/Documents/vich_prakt/polar coordinates
#im = Image.open('/home/nik/Documents/vich_prakt/polar coordinates/pic_color.jpg')
#im = Image.open('/home/nik/Desktop/test.tif')  # /home/nik/Documents/vich_prakt/polar coordinates

im = np.array(im)

H = im.shape[0]
W = im.shape[1]

im = padding(im,pix_r_MAX)

#temp_im = Image.fromarray(im.astype(np.uint8))
#temp_im.show()
#pause= input('pause')


im_res = np.zeros(im.shape)

r_Circle = 20

poin_i_0_of_cir = im.shape[0] // 2
 
poin_j_0_of_cir = im.shape[1] // 2


for i in range(pix_r_MAX ,im.shape[0]-pix_r_MAX):
    
    print(i, ' ---from--- ',im.shape[0]-pix_r_MAX)
    
    for j in range(pix_r_MAX,im.shape[1]-pix_r_MAX):
        
        temp_dec = dec(poin_i_0_of_cir,poin_j_0_of_cir,i,j)
        
        if temp_dec > r_Circle:
            
            temp_r = int(abs(temp_dec  - r_Circle) / 4 ) + 2
            
            if temp_r > pix_r_MAX:
                temp_r = pix_r_MAX
            
            #im_res[i,j,0],im_res[i,j,1],im_res[i,j,2] =GauB_1pix(im,i,j,temp_r,sigma)
            temp_small_im = im[i - temp_r : i + 1 + temp_r, j - temp_r : j + 1 + temp_r , : ]
            
            Weight_Matrix = Weight_Matrix_(sigma,temp_r)

            im_res[i,j,0],im_res[i,j,1],im_res[i,j,2] = int(np.sum(temp_small_im[:,:,0] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,1] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,2] * Weight_Matrix ))
        else:
            im_res[i,j] = im[i,j]
            
            
        
        



















#------------------------ OUTPUT ------------------------
new_im = Image.fromarray(im_res[pix_r_MAX:H+pix_r_MAX, pix_r_MAX:W+pix_r_MAX].astype(np.uint8))
new_im.save('/home/nik/Documents/Gaussian_Blur/pic_resss12_1.jpg')


#------------------------ OUTPUT ------------------------




















