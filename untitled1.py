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

def ext(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

def gauss_noise(image):
    h, w, ch = image.shape
    for row in range(h):
        for col in range(w):
        
        	# numpy.random.normal (loc, scale, size) Генерация случайных чисел плотности вероятности гауссова распределения
        	# loc: float представляет собой среднее значение сгенерированного случайным числом распределения Гаусса
        	# scale: float представляет дисперсию этого распределения
        	# size: int или кортеж формы вывода ints, по умолчанию None, выводится только одно значение 
        	# Если указано целое число, выводится целочисленное значение или (a, b) → строка и столбец b
            s = np.random.normal(0, 10, 3)
            # Удаляем три значения канала каждого пикселя
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            # Добавить гауссов шум к трем значениям канала каждого пикселя
            image[row, col, 0] = ext(b + s[0])
            image[row, col, 1] = ext(g + s[1])
            image[row, col, 2] = ext(r + s[2])
    cv.imshow("gauss_noise", image)


# gauss_noise(im)

s = np.random.normal(0, 1.5, im.shape[:-1])
#print(s)

for i in range (im.shape[0]):
    for j in range (im.shape[1]):
        im_res[i,j,0] = im[i, j, 0] * s[i,j] * im_res[i,j,0]
        im_res[i,j,1] = im[i, j, 1] * s[i,j] * im_res[i,j,0]
        im_res[i,j,2] = im[i, j, 2] * s[i,j] * im_res[i,j,0]
        
        



from matplotlib import pyplot as plt
plt.imshow(im_res, interpolation='nearest')
plt.show()



