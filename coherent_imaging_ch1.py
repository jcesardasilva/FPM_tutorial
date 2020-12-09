# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:18:14 2016

@author: jcesardasilva
"""

#from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera, astronaut
from skimage.transform import resize

# simulate a high-resolution object
objectIntensity = resize(np.double(camera()),(256,256), mode='reflect')
objectAmplitude = np.sqrt(objectIntensity)

# display object
plt.figure(1)
plt.imshow(objectAmplitude,cmap='bone')
plt.title('Input object (amplitude)')
plt.show()

# set up the parameters for the coherent imaging system
waveLength = 0.5e-6 # in m
k0 = 2*np.pi/waveLength
pixelSize = 0.5e-6 # in m
NA = 0.1
cutoffFrequency = NA*k0

# set up the low-pass filter
objectAmplitudeFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(objectAmplitude)))
m,n = objectAmplitude.shape
kx = np.arange(-np.pi/pixelSize,np.pi/pixelSize,2*np.pi/(n*pixelSize)) # 2*np.pi/(pixelSize*(n-1)))kmax/((n1)/2))
ky = np.arange(-np.pi/pixelSize,np.pi/pixelSize,2*np.pi/(m*pixelSize))
kxm,kym = np.meshgrid(kx,ky)
CTF = ((kxm**2+kym**2)<cutoffFrequency**2) # the coherent transfer function

# display the fourier space
plt.figure(2)
plt.imshow(CTF,cmap='bone')
plt.title('CTF in the spatial frequency domain')
plt.show()

# the filtering process
outputFT = CTF*objectAmplitudeFT

# display object
plt.figure(3)
plt.imshow(np.log(np.abs(outputFT)+0.5),cmap='jet')
plt.title('Filtered spectrum in the spatial frequency domain')
plt.show()

# output amplitude and intensity
outputAmplitude = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(outputFT)))
outputIntensity = np.abs(outputAmplitude)**2

# display object
plt.figure(4)
plt.imshow(outputIntensity,cmap='bone')
plt.title('Output object (amplitude)')
plt.show()
