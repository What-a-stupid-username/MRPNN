import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math

image1 = plt.imread(sys.argv[1], 0)
image2 = plt.imread(sys.argv[2], 0)
image3 = np.zeros((image1.shape[0],image1.shape[1],1), dtype=float)
image4 = np.zeros((image1.shape[0],image1.shape[1],1), dtype=float)
image5 = np.zeros((image1.shape[0],image1.shape[1],1), dtype=float)

mseall = 0.0
msecnt = 0
for i in range(image1.shape[0]):
    for j in range(image1.shape[1]):
        i0 = (image1[i,j,0]*0.299+image1[i,j,1]*0.587+image1[i,j,2]*0.114)/255.0
        i1 = (image2[i,j,0]*0.299+image2[i,j,1]*0.587+image2[i,j,2]*0.114)/255.0
        imgmse = (i0 - i1) * (i0 - i1)
        image1[i,j,0] = image1[i,j,1] = image1[i,j,2] = i0 * 255
        image2[i,j,0] = image2[i,j,1] = image2[i,j,2] = i1 * 255
        image3[i,j,0] = imgmse
        mseall = mseall + imgmse
        msecnt = msecnt + 1

mseavg = mseall / msecnt
mseavg = math.sqrt(mseavg)
print(mseavg * 100)
fig = plt.figure()
fig.add_subplot(221).imshow(image1)
plt.axis('off')
fig.add_subplot(222).imshow(image2)
plt.axis('off')
a = fig.add_subplot(223).imshow(image3,cmap = 'viridis',vmin=0.0, vmax=0.05)
plt.axis('off')
plt.title('RMSEx100: ' + str(mseavg * 100))
fig.colorbar(a)
plt.show()
