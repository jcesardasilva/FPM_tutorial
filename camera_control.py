import pygame
import pygame.camera
import time
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc


pygame.camera.init()
pygame.camera.list_cameras()
#cam = pygame.camera.Camera("/dev/video0", (640, 480))
cam = pygame.camera.Camera("/dev/video1", (1280, 720))
cam.start()
time.sleep(0.1)  # You might need something higher in the beginning

data = 0
total_expo = 0
for ii in range(100):
    print(ii)
    p0=time.time()
    img = cam.get_image()
    pf = time.time()-p0
    print('Exposure time: {}'.format(pf))
    total_expo += pf
    time.sleep(0.1)
    data += pygame.surfarray.array2d(img)
print('Total Exposure time: {} s'.format(total_expo))
plt.figure()
plt.imshow(data,cmap='gray')
plt.show(block=False)
pygame.image.save(img, "pygame.jpg")
#scipy.misc.toimage(image_array, cmin=0.0, cmax=...).save('outfile.jpg')
#im = Image.fromarray(data)
#im.save("pygame.jpeg")
cam.stop()

### use openCV because the number of counts for the surface pygame is very high
import time
import cv2
camera_port = 1#0
camera = cv2.VideoCapture(camera_port)
time.sleep(0.1)  # If you don't wait, the image will be dark
camera.read() # just to remove overhead
time.sleep(0.1)
data = 0
total_expo = 0
for ii in range(100):
    print(ii)
    p0=time.time()
    return_value, image = camera.read()
    pf = time.time()-p0
    print('Exposure time: {}'.format(pf))
    total_expo += pf
    time.sleep(0.1)
    data += image.astype(int32)
print('Total Exposure time: {} s'.format(total_expo))
plt.figure()
#plt.imshow(image,cmap='gray')
plt.imshow(data.mean(2),cmap='gray')
plt.show(block=False)
cv2.imwrite("opencv.jpeg", image)#image)
del(camera)  # so that others can use the camera as soon as possible
