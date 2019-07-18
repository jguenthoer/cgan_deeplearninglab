# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:33:43 2019

@author: Jan
implements a class batchloader
loads images from a given directory
splits them in training, test and validation sets
gives out randomized batches
"""

import cv2
import glob
import numpy as np
import random
import csv


class BatchLoader():
    def __init__(self, path):
        self.path = path
        self.images = glob.glob(path+'/img_align_celeba/*jpg') # list of all images at path
        numimages = len(self.images)
        print(path)
        print(type(self.images))
        #split into training, test and validation sets
        #default split 80/15/5
        training = int(numimages*0.8)
        test = int(numimages*0.95)
        self.trainingset = self.images[:training]
        self.testset = self.images[training:test]
        self.validationset = self.images[test:]
        
        # read in attributes
        f = open(path+"/list_attr_celeba.txt")
        reader = csv.reader(f, delimiter = ' ')
        self.attributes = {}
        for idx, row in enumerate(reader):
            
           #ignore first row
           if idx != 0 and idx != 1:
               row = list(filter(None, row))
               attr = np.array(row[1:], dtype=int)
               attr = (attr > 0)*1
               self.attributes[row[0]] = attr
           
        f.close()
        # read in landmarks
        f = open(path+"/list_landmarks_align_celeba.txt")
        reader = csv.reader(f, delimiter = ' ')
        self.landmarks = {}
        for idx, row in enumerate(reader):
            
           #ignore first row
           if idx != 0 and idx != 1:
               row = list(filter(None, row))
               landmk = np.array(row[1:], dtype = np.float32)
               landmk /= 255 
               self.landmarks[row[0]] = landmk
               
        f.close()
        
    def gettrainingbatch(self, batchsize):

        #shuffle the set
        samples = self.trainingset
        random.shuffle(samples)
        numbatch = (len(samples)//batchsize)
        
        #create output
        #dimensions: numberofbatchesxbatchsizex(3x128x128)
        i = 0
        while i < numbatch:
           impaths = samples[i*batchsize:((i+1)*batchsize)]
           batch = np.empty((batchsize, 3, 218, 178), dtype = np.uint8)
           att = np.empty((batchsize, 40), dtype = int)
           landmarks = np.empty((batchsize, 10))
           for j,path in enumerate(impaths):
               image = cv2.imread(path)
               # get axis in right order for the CNN (128x128x3 -> 3x128x128)
               batch[j] = np.moveaxis(image, 2, 0)
               # match attributes with images
               key = path.split('/')[-1]
               att[j] = self.attributes[key]
               landmarks[j] = self.landmarks[key]
               
           yield batch, att, landmarks
           i += 1
           
        

                      
            
        
    def gettestbatch(self, batchsize):

        samples = self.testset
        random.shuffle(samples)
        numbatch = (len(samples)//batchsize)
        
        #create output
        #dimensions: numberofbatchesxbatchsizex(3x128x128)
        i = 0
        while i < numbatch:
           impaths = samples[i*batchsize:((i+1)*batchsize)]
           batch = np.empty((batchsize, 3, 218, 178), dtype = np.uint8)
           att = np.empty((batchsize, 40), dtype = int)
           landmarks = np.empty((batchsize, 10))
           for j,path in enumerate(impaths):
               image = cv2.imread(path)
               # get axis in right order for the CNN (128x128x3 -> 3x128x128)
               batch[j] = np.moveaxis(image, 2, 0)
               # match attributes with images
               key = path.split('/')[-1]
               att[j] = self.attributes[key]
               landmarks[j] = self.landmarks[key]
               
           yield batch, att,landmarks
           i += 1
                  
    def getvalidationbatch(self, batchsize):

        samples = self.validationset
        random.shuffle(samples)
        numbatch = (len(samples)//batchsize)
        
        #create output
        #dimensions: numberofbatchesxbatchsizex(3x128x128)
        i = 0
        while i < numbatch:
           impaths = samples[i*batchsize:((i+1)*batchsize)]
           batch = np.empty((batchsize, 3, 218, 178), dtype = np.uint8)
           att = np.empty((batchsize, 40), dtype = int)
           landmarks = np.empty((batchsize, 10))
           for j,path in enumerate(impaths):
               image = cv2.imread(path)
               # get axis in right order for the CNN (128x128x3 -> 3x128x128)
               batch[j] = np.moveaxis(image, 2, 0)
               # match attributes with images
               key = path.split('/')[-1]
               att[j] = self.attributes[key]
               landmarks[j] = self.landmarks[key]
               
           yield batch, att, landmarks
           i += 1
        