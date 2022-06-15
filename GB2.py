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

def gauss_func(sigma_, pos): # Gaussian function (sigma_ - sigma ; pos - coordinate  in Gaussian function)
    return (1 / ( (2 * np.pi * sigma_ ** 2) ** 0.5 ) ) * np.e ** (-((pos ** 2)/(2  * sigma_ ** 2)))

#------------------------ LOAD IMAGE ------------------------
im = Image.open('/home/nik/Documents/Gaussian_Blur/pic.jpg')
im = np.array(im)
im_res = im.copy()





pos_x = np.array([[-1,0,1],
                 [-1,0,1],
                 [-1,0,1]],dtype= np.int32)

pos_y = np.array([[1,1,1],
                 [0,0,0],
                 [-1,-1,-1]],dtype= np.int32)


Weight_Matrix = gauss_func(sigma,pos_x) * gauss_func(sigma,pos_y)



    

# А теперь в чем иедя. Для КАЖДОЙ точки надо взять ее соседей.
# И показания по RGB  умножить на весовую матрицу (Weight_Matrix)
# после чего полученное значение надо сложить непосредственно с центральным 
# TODO 
#
#
#
#


