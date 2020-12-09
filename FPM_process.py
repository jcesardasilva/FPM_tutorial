#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:18:14 2016

@author: jcesardasilva
"""
import numpy as np
import matplotlib.pyplot as plt
#from skimage.data import camera, astronaut
import subprocess 
import os, time, re, glob

if __name__=="__main__":
    first_frame = 'image1/image1_0001.pgm'
    led_pos_filename = 'image1/led_pos.txt'
    #resolution = '1280x720'
    #file_prefix = 'image1'
    
    abspath = os.path.abspath(first_frame)
    filename = os.path.basename(first_frame)
    body,ext = os.path.splitext(abspath)
    file_wcard = ''.join((re.sub('\d{4}$','*',body),ext))

    file_list = sorted(glob.glob(file_wcard))

    # read the first file to get dimensions
    img0 = plt.imread(file_list[0])
    r,c = img0.shape
    img_array = np.empty((len(file_list),r,c))
    pos_array = np.empty((len(file_list)))

    for idx,ii in enumerate(file_list):
        print('Reading file {}'.format(ii))
        img_array[idx] = plt.imread(ii).astype(np.float)
    
    
    with open(led_pos_filename,'r') as fid:
        for ii in range(len(file_list)):        
            pos_array[ii] = eval(fid.readline())

