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

# set up the coherent transfer function in the Fourier domain
objectIntensityFT = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(objectIntensity)))
m,n = objectIntensity.shape
kx = np.arange(-np.pi/pixelSize,np.pi/pixelSize,2*np.pi/(n*pixelSize)) # 2*np.pi/(pixelSize*(n-1)))kmax/((n1)/2))
ky = np.arange(-np.pi/pixelSize,np.pi/pixelSize,2*np.pi/(m*pixelSize))
kxm,kym = np.meshgrid(kx,ky)
CTF = ((kxm**2+kym**2)<cutoffFrequency**2) # pupil function circ (max)

# display the fourier space
plt.figure(2)
plt.imshow(CTF,cmap='bone')
plt.title('CTF in the spatial frequency domain')
plt.show()

# the set up the incoherent transfer function
cpsf = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(CTF))) # coherent psf
ipsf = np.abs(cpsf)**2 # incoherent psf
OTF = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ipsf))))
OTF /= OTF.max()

# display the incoherent transfer function in the Fourier domain
plt.figure(3)
plt.imshow(np.abs(OTF),cmap='bone')
plt.title('Incoherent transfer function in the Fourier domain')
plt.show()

# perform low-pass filtering and generate the output intensity image
outputFT = OTF*objectIntensityFT

# Filtered spectrum in the Fourier Space
plt.figure(4)
plt.imshow(np.log(np.abs(objectIntensityFT)+0.5),cmap='jet')
plt.title('Filtered spectrum in the Fourier Space')
plt.show()

outputIntensity = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(outputFT))).real

# display object Intensity
plt.figure(5)
plt.imshow(outputIntensity,cmap='bone')
plt.title('Output object (intensity)')
plt.show()
