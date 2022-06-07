#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:03:20 2022

@author: nik

"""

import math
from PIL import Image


#https://www.pixelstech.net/article/1353768112-Gaussian-Blur-Algorithm

def gauss_func(stdev, pos):
    return (1 / ( (2 * math.pi * stdev ** 2) ** 0.5 ) ) * math.e ** (-((pos ** 2)/(2  *stdev ** 2)))


def is_valid_pos(max_width,max_height,pos_x,pos_y): #TODO position possibility (border check)
    return (0 < pos_x < max_width) and (0 < pos_y < max_height)


def reSize(img_, save_): #TODO reSize  pic (.jpg) and may save them
    
    width = img_.width // 3
    height = img_.height // 3
    
    resized_img_ = img_.resize((width, height), Image.ANTIALIAS)
    
    if save_ != 0:
        resized_img_.save("resized_img.jpg")
    return resized_img_



#image_file = input("Input file name:\n")
im = Image.open('/home/nik/Documents/Gaussian_Blur/pic.jpg')
#im = reSize(im,0) #TODO resize the image to make  run faster 
im_res = im.copy() #TODO create a copy of pic

SIZE_X = im.width
SIZE_Y = im.height
################## INPUT ##################
pix_r = 5 
sigma = 1.5

box_offsets = [] #TODO generates all offset combos


for i in range(-pix_r,pix_r+1): #TODO generation of possible variations 
    for j in range(-pix_r,pix_r+1):
        box_offsets.append((i,j))

#create Weight matrix
coeff = [gauss_func(sigma, x)*gauss_func(sigma, y) for x,y in box_offsets]

#a = input("pause")



for x_val in range(SIZE_X):
    
    for y_val in range(SIZE_Y):
        
        res_r = 0
        res_b = 0
        res_g = 0
        
        i = 0
        
        for val in box_offsets:
            
            if is_valid_pos(SIZE_X, SIZE_Y, x_val + val[0], y_val + val[1]):
                pixel_r, pixel_b, pixel_g = im.getpixel((x_val + val[0], y_val + val[1]))
            else:
                pixel_r, pixel_b, pixel_g = im.getpixel((x_val,y_val))
            
            res_r += (pixel_r * coeff[i])
            res_b += (pixel_b * coeff[i])
            res_g += (pixel_g * coeff[i])
            i += 1
            
        im_res.putpixel((x_val,y_val),(int(res_r),int(res_b),int(res_g)))

im_res.show()

