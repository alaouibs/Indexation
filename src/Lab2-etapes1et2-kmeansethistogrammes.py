#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

import os
import pandas as pd
import pickle

import csv
import copy


# In[50]:


def getSurf(img):
    surf = cv.xfeatures2d.SURF_create(5000)
    kp, des = surf.detectAndCompute(img,None)
    return kp, des

def getSift(img):
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    kp,des = sift.compute(gray,kp)
    return kp, des


# In[51]:


descriptors_images = []
descriptors2 = []
for filename in os.listdir('../data/jpg3'):
    img = cv.imread(os.path.join('../data/jpg3',filename))
    if img is not None:
        print(filename)
        kp,des = getSurf(img)
        descriptors_images.append((filename, des))


# In[52]:


alldes = []
alldes_filename = []
for filename, des in descriptors_images:
    for item in des:
        alldes.append(item)
        alldes_filename.append(filename)
alldes = np.array(alldes)
alldes_filename = np.array(alldes_filename)


# In[53]:


alldes_filename


# In[ ]:


k = 100
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,cente = cv.kmeans(alldes, k,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)


# In[ ]:


label


# In[ ]:


lab = []
for l in label:
    lab.append(l[0])
    
lab = np.array(lab)


# In[ ]:


label.flatten()


# In[ ]:


df = pd.DataFrame(np.vstack((alldes_filename, lab)).T, columns=['image', 'cluster']) 


# In[ ]:


df.to_csv("../data/clusters.csv", encoding='utf-8', index=False)


# In[ ]:




