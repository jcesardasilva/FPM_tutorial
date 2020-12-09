# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 2017

@author: jcesardasilva
"""

#from __future__ import print_function, division
import numpy as np

## auxiliary function
def gseq(arraysize):
    n = (arraysize+1)/2. # redundant with line below
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
    seqf[0,0:arraysize**2] = seq-1
    return seqf.astype(np.int)
