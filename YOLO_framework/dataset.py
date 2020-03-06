import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import cv2



def match(ann_box,ann_confidence,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [5,5,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [5,5,number_of_classes], ground truth class labels to be updated
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    size = 5 #the size of the output grid
    
    #update ann_box and ann_confidence
    # x
    center_x = (x_min+x_max)/2
    cell_x = int(center_x//0.2)
    rele_center_x = (center_x-0.2*cell_x)/0.2
    sqr_width = np.sqrt(x_max-x_min)
    # y
    center_y = (y_min+y_max)/2
    cell_y = int(center_y//0.2)
    rele_center_y = (center_y-0.2*cell_y)/0.2
    sqr_height = np.sqrt(y_max-y_min)

    ann_box[cell_y,cell_x,0] = rele_center_x
    ann_box[cell_y,cell_x,1] = rele_center_y
    ann_box[cell_y,cell_x,2] = sqr_width
    ann_box[cell_y,cell_x,3] = sqr_height
    ann_confidence[cell_y,cell_x,cat_id] = 1
    ann_confidence[cell_y,cell_x,-1] = 0
    return ann_box,ann_confidence
    
    
    
class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        #notice:
        #you can split the dataset into 80% training and 20% testing here, by slicing self.img_names with respect to self.train
        offset = int(len(self.img_names)*0.8)
        if self.train:
            self.img_names = self.img_names[:offset]
        else:
            self.img_names = self.img_names[offset:]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        size = 5 #the size of the output grid
        ann_box = np.zeros([5,5,4], np.float32)#5*5 bounding boxes
        ann_confidence = np.zeros([5,5,self.class_num], np.float32) #5*5 one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,:,-1] = 1 #the default class for all cells is set to "background"
    
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:

        #1. prepare the image [3,320,320], by reading image "img_name" first.
        image = cv2.imread(img_name)
        x_shape = image.shape[1]
        y_shape = image.shape[0]
        image = cv2.resize(image,(320,320))
        image = np.transpose(image,(2,0,1))
        x_scale = 320/x_shape
        y_scale = 320/y_shape
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        anno_txt = open(ann_name)
        anno = anno_txt.readlines()
        anno_txt.close()
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        for i in range(len(anno)):
            line = anno[i].split()
            class_id = int(line[0])
            x_start = float(line[1])*x_scale
            y_start = float(line[2])*y_scale
            w = float(line[3])*x_scale
            h = float(line[4])*y_scale
            #color = colors[c]
            #img_train = cv2.rectangle(img,(start_x,start_y),(start_x+w,start_y+h),color,2)
            #crop_img = img_train[start_y+2:start_y+h-2, start_x+2:start_x+w-2]
            x_min = x_start/320
            y_min = y_start/320
            x_max = (x_start+w)/320
            y_max = (y_start+h)/320
            ann_box,ann_confidence = match(ann_box,ann_confidence,class_id,x_min,y_min,x_max,y_max)
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.

        #to use function "match":
        #match(ann_box,ann_confidence,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        #you may wonder maybe it is better to input [x_center, y_center, box_width, box_height].
        #maybe it is better.
        #BUT please do not change the inputs.
        #Because you will need to input [x_min,y_min,x_max,y_max] for SSD.
        #It is better to keep function inputs consistent.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
        
        return image, ann_box, ann_confidence
