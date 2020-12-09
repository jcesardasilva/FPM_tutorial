#!/usr/bin/python3
# -*- coding: utf-8 -*-

# standard python package
import itertools
import time

# third party package
import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
from skimage.io import imread, use_plugin
from skimage.transform import resize

__all__=[
    "load_pngimg",
    "generateseq",
    "generate_kvectors",
    "generate_CTF",    
]

def load_pngimg(filename):
    """
    Read image from PNG file
    """
    use_plugin('pil')
    return imread(filename)
    
def generateseq(arraysize):
    """
    Auxiliary function to sort positions starting from the center
    """
    n = np.ceil(arraysize/2.)#
    #n = (arraysize+1)/2. # redundant with line below
    #arr_size = 2*n-1
    sequence = np.zeros((2,arraysize**2))
    sequence[0,0]=n
    sequence[1,0]=n
    dx =+1
    dy =-1
    stepx =+1
    stepy =-1
    direction =+1
    counter = 0
    for ii in range(1,arraysize**2):
        counter +=1
        if (direction==+1):
            sequence[0,ii] = sequence[0,ii-1]+dx
            sequence[1,ii] = sequence[1,ii-1]
            if (counter == np.abs(stepx)):
                counter = 0
                direction *= -1
                dx *= -1
                stepx *= -1
                if stepx>0:
                    stepx += 1
                else:
                    stepx -= 1
        else:
            sequence[0,ii] = sequence[0,ii-1]
            sequence[1,ii] = sequence[1,ii-1]+dy
            if (counter == np.abs(stepy)):
                counter = 0
                direction *= -1
                dy *= -1
                stepy *= -1
                if stepy>0:
                    stepy += 1
                else:
                    stepy -= 1
    seq = (sequence[0,:]-1)*arraysize+sequence[1,:]
    seqf = np.zeros((1,arraysize**2))
    if arraysize%2: # if arraysize is odd => mod=1
        seqf[0,0:arraysize**2] = seq-1
    else:
        seqf[0,0:arraysize**2] = seq
    return seqf.astype(np.int)
    
def generate_kvectors(arraysize,LEDgap,LEDheight):
    """"
    Create the wave vectors for the LED illumination
    """
    x = np.arange(arraysize) - np.arange(arraysize).mean()#np.arange(-arraysize/2,arraysize/2)
    y = x.copy() # suppose square array
    x*=LEDgap
    y*=LEDgap
    
    xy = np.array(list(itertools.product(x,y)))
    xlocation = xy[:,0]
    ylocation = xy[:,1]
    
    kx_relative = -np.sin(np.arctan(xlocation/LEDheight)) # create kx, ky wavevectors
    ky_relative = -np.sin(np.arctan(ylocation/LEDheight))
        
    return kx_relative, ky_relative

def generate_CTF(obj_shape,arraysize,waveLength,spsize,psize,NA,LEDgap,LEDheight):
    """
    Generate the coherent transfer function
    """
    ## setup the parameters for the coherent imaging system
    m,n = obj_shape # image size of the high resolution object
    resolution_factor = spsize/psize# spsize/psize = 4
    m1 = int(m/resolution_factor) # spsize/psize = 4
    n1 = int(n/resolution_factor) # image size of the final output
    k0 = 2*np.pi/waveLength
    kx_relative,ky_relative = generate_kvectors(arraysize,LEDgap,LEDheight)
    kx = k0 * kx_relative # (1,225)
    ky = k0 * ky_relative # (1,225)
    dkx = 2*np.pi/(psize*n) # sampling in the fourier space (psize*n = Lx)
    dky = 2*np.pi/(psize*m) # (psize*m = Ly)
    cutoffFrequency = NA * k0
    kmax = np.pi/spsize # Nyquist frequency using the pixel size in the CCD (image)

    # designing the coherent transfer function
    kxm = np.arange(-kmax,kmax,kmax/(n1/2)) #kmax/(n1/2) = 2*np.pi/(n1*spsize))
    kym = np.arange(-kmax,kmax,kmax/(m1/2)) #kmax/(m1/2) = 2*np.pi/(m1*spsize))
    Kxm, Kym = np.meshgrid(kxm,kym)
    CTF = (Kxm**2 + Kym**2)<cutoffFrequency**2 # coherent transfer function (64 x 64)
    
    return CTF,kx,dkx,ky,dky

