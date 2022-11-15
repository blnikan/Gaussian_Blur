#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:04:09 2022

@author: nik
@author git: https://github.com/blnikan

"""

import cv2
import numpy  as np


def gauss_func(sigma_, pos): # Gaussian function (sigma_ - sigma ; pos - coordinate  in Gaussian function)
    return (1 / ( (2 * np.pi * sigma_ ** 2) ** 0.5 ) ) * np.e ** (-((pos ** 2)/(2  * sigma_ ** 2)))

def Weight_Matrix_xz(sigma_, rad_):#???
    
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


def gB_zone(img_,r_,Weight_Matrix_):
    H_ = img_.shape[0]
    W_ = img_.shape[1]
    img_ = padding(img_,r_)
    
    i_size = img_.shape[0]
    j_size = img_.shape[1]
    
    im_res_ = np.zeros([i_size, j_size,3])
    
    for i in range(i_size):
        for j in range(j_size):
            temp_small_im = img_[i - r_ : i + 1 + r_, j - r_ : j + 1 + r_ , : ]
    
            im_res_[i,j,0],im_res_[i,j,1],im_res_[i,j,2] = int(np.sum(temp_small_im[:,:,0] * Weight_Matrix_ )),int(np.sum(temp_small_im[:,:,1] * Weight_Matrix_ )),int(np.sum(temp_small_im[:,:,2] * Weight_Matrix_ ))
    
    return im_res_[r_:H_+r_, r_:W_+r_]

def sea_zone(img1, img2,r_,Weight_Matrix_,dead):
    """
    img1-начальное изображение
    img2-новое изображение
    
    О---------О
    | 1  | 2  |
    |    |    |   
    О---------О
    |  3 |  4 |
    |    |    |
    О---------О   
    """
    i_dev_2 = img1.shape[0]/2
    j_dev_2 = img1.shape[1]/2
    
    img_1_1 = img1[0:i_dev_2,0:j_dev_2 ]
    img_1_2 = img1[0:i_dev_2,j_dev_2: ]
    img_1_3 = img1[i_dev_2:,0:j_dev_2 ]
    img_1_4 = img1[i_dev_2:,j_dev_2: ]

    img_2_1 = img2[0:i_dev_2,0:j_dev_2 ]
    img_2_2 = img2[0:i_dev_2,j_dev_2: ]
    img_2_3 = img2[i_dev_2:,0:j_dev_2 ]
    img_2_4 = img2[i_dev_2:,j_dev_2: ]
    
    if dead == 26:
        img_2_1 = gB_zone(img_2_1)
        img_2_2 = gB_zone(img_2_2)
        img_2_3 = gB_zone(img_2_3)
        img_2_4 = gB_zone(img_2_4)
    
    
    if np.sum(np.equal(img_1_1,img_2_1)) == i_dev_2*j_dev_2*4:
        img_2_1 = img_1_1
    else:
        sea_zone(img_1_1,img_2_1,r_,Weight_Matrix_,dead+1)
    
    if np.sum(np.equal(img_1_2,img_2_2)) == i_dev_2*j_dev_2*4:
        img_2_2 = img_1_2
    else:
        sea_zone(img_1_2,img_2_2,r_,Weight_Matrix_,dead+1)
    
    if np.sum(np.equal(img_1_3,img_2_3)) == i_dev_2*j_dev_2*4:
        img_2_3 = img_1_3
    else:
        sea_zone(img_1_3,img_2_3,r_,Weight_Matrix_,dead+1)
    
    if np.sum(np.equal(img_1_4,img_2_4)) == i_dev_2*j_dev_2*4:
        img_2_4 = img_1_4
    else:
        sea_zone(img_1_4,img_2_4,r_,Weight_Matrix_,dead+1)
    
    img2[0:i_dev_2,0:j_dev_2 ] = img_2_1
    img2[0:i_dev_2,j_dev_2: ]  = img_2_2
    img2[i_dev_2:,0:j_dev_2 ]  = img_2_3
    img2[i_dev_2:,j_dev_2: ]   = img_2_4
    
    
    return img2
    
    
    
    
    
vid = cv2.VideoCapture('/home/nik/Documents/Gaussian_Blur/lin.mp4')

#pix_r = 7
pix_r_MAX = pix_r
sigma = 3

if (vid.isOpened()== False): 
  print("Error opening video stream or file")

frames = []   

cou = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/nik/Documents/Gaussian_Blur/lin_out.avi',fourcc, 20.0, (600,600))

while(vid.isOpened()):
    ret, frame = vid.read()
    
    if ret:
        print(cou)
        if cou%2 == 0:
            
            """
            H = frame.shape[0]
            W = frame.shape[1]
    
            frame = padding(frame,pix_r)
            
            im_res =  np.zeros(frame.shape)
            
            Weight_Matrix = Weight_Matrix_(sigma,pix_r)
            
            for i in range (pix_r,frame.shape[0] - pix_r-1):
                #print(f'i: {i} from im.shape[0] - pix_r')
    
                for j in range (pix_r,frame.shape[1] - pix_r-1):
                    #print(f'j---{j}')
                    #0-rojo (red)
                    #1-verde (green)
                    #2-azul (blue)
    
                    temp_small_im = frame[i - pix_r : i + 1 + pix_r, j - pix_r : j + 1 + pix_r , : ]
                    frame[i,j,0],frame[i,j,1],frame[i,j,2] = int(np.sum(temp_small_im[:,:,0] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,1] * Weight_Matrix )),int(np.sum(temp_small_im[:,:,2] * Weight_Matrix ))
            """
        cou+=1
        #cv2.imwrite(f'/home/nik/Documents/Gaussian_Blur/Torvalds3{cou}.jpg', frame)  
        #frame = np.dtype(frame,np.uint8)        
        
        out.write(frame.astype(np.uint8))
        frames.append(frame)
        
        if cou == 16:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
        	
            break
        


# When everything done, release the video capture object
vid.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()