import numpy as np
import coloredlogs,logging

coloredlogs.install()

#wavelen = 637e-9
wavelen = 632e-9
#wavelen = 625e-9
mag = 4
NA = 0.1
D_led = 4e-3#3.5e-3
pd = 5.5e-6 #3.2e-6
h = 4.4e-2#3.5e-2#2.7e-2#1.5e-2
nLeds = 64

#~ wavelen = 630e-9
#~ mag = 2
#~ NA = 0.08
#~ D_led = 4e-3
#~ pd = 2.75e-6 #3.2e-6
#~ h = 90e-3
#~ nLeds = 15**2

ps = mag*wavelen/(2*NA)
print('The sampling pixel should be smaller than {:.02e} m'.format(ps))
f_obj = NA/wavelen
print('The frequency cutoff of the objective is {:02.02f}'.format(f_obj))
f_cam = mag/(2*pd)
print('The frequency cutoff defined by the camera is {:02.02f}'.format(f_cam))
f_led = (1/wavelen)*(D_led/(np.sqrt(D_led**2+h**2)))
print('The minimum tilted illumination frequency is {:02.02f}'.format(f_led))

R_cam = f_cam/f_obj
print('\nR_cam is {:02.02f}'.format(R_cam))
if R_cam > 1:
    print('The sampling in the spatial domain is right')
else:
    logging.warning('The sampling in the spatial domain is not right')

R_led = f_obj/f_led # the same as NA*(np.sqrt(D_led**2+h**2)/D_led)
print('\nR_led is {:02.02f}'.format(R_led))
if R_led > 0.5:
    print('The sampling in the frequency domain is right')
    R_overlap = (1/np.pi)*((2*np.arccos(1/(2*R_led)))- ((1/R_led)*( np.sqrt(1-(1/(2*R_led))**2) )) )
    print('The aperture-overlapping-rate is {:02.02f}'.format(R_overlap))
else:
    logging.warning('The sampling in the frequency domain is not right')

halfLengh = np.sqrt(nLeds)//2
f_led_max = (1/wavelen)*(halfLengh*D_led/(np.sqrt((halfLengh*D_led)**2+h**2)))
# equivalent to :
# k0 = (2*np.pi)/wavelen
# kmax = k0*(4*D_led/(np.sqrt((4*D_led)**2+h**2)))
# NAillu = kmax/k0  #### -> n np.sin(theta_max)
NA_illu = f_led_max*wavelen
NA_syn= NA_illu + NA # Should I add NA or not??
print('\nThe synthetic NA is {:.2f}'.format(NA_syn))

p_max_recon = wavelen/(4*NA_syn)
print('The largest pixel size in the reconstruction is {:.2f} nm'.format(p_max_recon*1e9))

enhancement_factor = 2*NA_syn/NA
print('The enhancement factor is {:.2f}'.format(enhancement_factor))

