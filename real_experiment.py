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
import os, time

def cam_acquisition(filename,device='/dev/video1',resolution ='1280x720'):
    #os.system('fswebcam --no-banner -d {} -F 1 --greyscale -r {} {}.jpg'.format(device,resolution,filename))
    #subprocess.call('fswebcam --no-banner -d {} -F 1 --greyscale -r {} {}.jpg'.format(device,resolution,filename).split())
    flag=subprocess.call('ffmpeg -y -v error -f video4linux2 -i {} -s {} -frames 1 {}'.format(device,resolution,filename).split())
    return flag
'''FFMPEG loglevel
const struct { const char *name; int level; } log_levels[] = {
        { "quiet"  , AV_LOG_QUIET   },
        { "panic"  , AV_LOG_PANIC   },
        { "fatal"  , AV_LOG_FATAL   },
        { "error"  , AV_LOG_ERROR   },
        { "warning", AV_LOG_WARNING },
        { "info"   , AV_LOG_INFO    },
        { "verbose", AV_LOG_VERBOSE },
        { "debug"  , AV_LOG_DEBUG   },
    };
'''
if __name__=="__main__":
    camera = '/dev/video1'
    resolution = '1280x720'
    file_prefix = 'image1'
    number_exposures = 2
    file_extension = 'pgm'

    # create folder to stores images
    if not os.path.isdir(file_prefix):
        print('Creating folder {}'.format(file_prefix))
        os.makedirs(file_prefix)
    else:
        print('Folder {} already exists and will be used'.format(file_prefix))
    if not os.path.isfile('{}/led_pos.txt'.format(file_prefix)):
        print('Creating led_pos.txt file')
    else:
        print('File led_pos.txt already exists and will be overwritten')
        os.remove('{}/led_pos.txt'.format(file_prefix))
    
    total_expo = 0
    for ii in range(number_exposures):
        #filename = '_'.join((os.path.join(file_prefix,file_prefix),str(ii)))
        filename = './{}/{}_{:04d}.{}'.format(file_prefix,file_prefix,ii,file_extension)
        print('\nAcquisition {}: {}'.format(ii+1,filename))
        if os.path.isfile(filename):
            print('File already exists and will be overwritten')
        print('-------------------------------')
        p0=time.time()
        flag = cam_acquisition(filename,camera,resolution)
        if flag !=0:
            raise FileIO('The frame {} could not be acquired'.format(filename))
        pf = time.time()-p0
        print('Exposure time: {:.4f}s '.format(pf))
        total_expo += pf
        fid=open('{}/led_pos.txt'.format(file_prefix),'a');
        fid.write('{:.4e}\n'.format(ii))
        fid.close()
    print('Total Exposure time: {:.4f} s'.format(total_expo))
