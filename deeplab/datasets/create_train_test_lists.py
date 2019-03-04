'''
Author:     DT

Updated by: 

	This script works with dataset images( folders JPEGImage, ImageSets) to obtain data items and create shuffled  		train, val and train_val lists.
	They will be used by neural network in further traing task
'''



import os,sys
import numpy as np
from sklearn.model_selection import train_test_split
         
work_path      = "./WZL_rupture_seg/WZL_rupture_dataset/Rupture/"
images         = "JPEGImages"
imageSets      = "ImageSets/Segmentation/"
train_file     = "train.txt"
val_file       = "val.txt"
train_val_file = "trainval.txt"
  
if not os.path.exists(work_path + imageSets):
    os.makedirs(work_path + imageSets)

print("Create a folder with train, val , train_val lists")
     
try:    
   names = [os.path.splitext(filename)[0] for filename in os.listdir(work_path + images)]
except:
   print("------Problem with file access or opening-------")

train, val = train_test_split(names,test_size = 0.2, random_state = 42, shuffle=True)
train_val = train + val

try:
   with open(work_path + imageSets + train_file,"w") as f:
        for name in train:
            f.write("%s\n" % name)
   
   with open(work_path + imageSets + val_file,"w") as f:
        for name in val:
            f.write("%s\n" % name)
   with open(work_path + imageSets + train_val_file,"w") as f:
        for name in train_val:
            f.write("%s\n" % name)
except:
   print("There is a problem with file creating or writing")



