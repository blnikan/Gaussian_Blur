#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:03:20 2022

@author: nik

"""

import math
from PIL import Image

PI = math.pi
E = math.e

#https://www.pixelstech.net/article/1353768112-Gaussian-Blur-Algorithm

def gauss_func(stdev, pos):
    return (1 / ( (2 * math.pi * stdev ** 2) ** 0.5 ) ) * math.e ** (-((pos ** 2)/(2  *stdev ** 2)))


def is_valid_pos(max_width,max_height,pos_x,pos_y):
    return (0 < pos_x < max_width) and (0 < pos_y < max_height)


def reSize(img_, save_): # reSize  pic (.jpg) and may save them
    
    width = img_.width // 2
    height = img_.height // 2
    
    resized_img_ = img_.resize((width, height), Image.ANTIALIAS)
    
    if save_ == 0:
        return resized_img_
    else:
        resized_img_.save("resized_img.jpg")
        return resized_img_


#image_file = input("Input file name:\n")
im = Image.open('/home/nik/Documents/Gaussian_Blur/pic.jpg')
im = reSize(im,0) #reSize pic
im_res = im.copy() # create a copy of pic

MAX_X = im.width
MAX_Y = im.height

pix_r = 5#int(input("Input the blur radius:  "))
sigma = 1.5#int(input("Input the Gaussian standard deviation:  "))

box_offsets = [] #generates all offset combos

for i in range(-pix_r,pix_r+1):
    
    for j in range(-pix_r,pix_r+1):
        
        box_offsets.append((i,j))


coeff = [gauss_func(sigma, x)*gauss_func(sigma, y) for x,y in box_offsets]




for x_val in range(MAX_X):
    
    for y_val in range(MAX_Y):
        
        tot_r = 0
        tot_b = 0
        tot_g = 0
        
        for idx, valies in enumerate(box_offsets):
            
            if is_valid_pos(MAX_X, MAX_Y, x_val + valies[0], y_val + valies[1]):
                pixel_r, pixel_b, pixel_g = im.getpixel((x_val+valies[0],y_val+valies[1]))
            else:
                pixel_r, pixel_b, pixel_g = im.getpixel((x_val,y_val))
            tot_r += (pixel_r * coeff[idx])
            tot_b += (pixel_b * coeff[idx])
            tot_g += (pixel_g * coeff[idx])
        im_res.putpixel((x_val,y_val),(int(tot_r),int(tot_b),int(tot_g)))

im_res.show()
im_res.save("resized_img.jpg")
