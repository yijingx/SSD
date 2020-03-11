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
import random

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    boxes = np.zeros([135,4,8], np.float32)
    layer_idx = 0
    for i,grid_num in enumerate(layers): #10,5,3,1
        ssize = small_scale[i] #box1,w,h
        lsize = large_scale[i] #box2,w,h
        lsize_l = lsize*np.sqrt(2) #box3,h box4,w
        lsize_s = lsize/np.sqrt(2) #box4,w box3,h
        for center_idx in range(grid_num*grid_num):
            offset = 1/(grid_num*2)
            x_cell_idx = center_idx%grid_num
            y_cell_idx = center_idx//grid_num
            x_center = x_cell_idx/(grid_num)+offset #para_1
            y_center = y_cell_idx/(grid_num)+offset #para_2
            boxes[layer_idx,0] = [x_center, y_center, ssize, ssize, x_center-ssize/2, y_center-ssize/2, x_center+ssize/2, y_center+ssize/2]
            boxes[layer_idx,1] = [x_center, y_center, lsize, lsize, x_center-lsize/2, y_center-lsize/2, x_center+lsize/2, y_center+lsize/2]
            boxes[layer_idx,2] = [x_center, y_center, lsize_l, lsize_s, x_center-lsize_l/2, y_center-lsize_s/2, x_center+lsize_l/2, y_center+lsize_s/2]
            boxes[layer_idx,3] = [x_center, y_center, lsize_s, lsize_l, x_center-lsize_s/2, y_center-lsize_l/2, x_center+lsize_s/2, y_center+lsize_l/2]
            layer_idx += 1
    boxes = boxes.reshape(-1,8)
    boxes = np.clip(boxes,0,1)
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)

def iou_NMS(boxs_, highest_idx,boxes_default):
    #input:
    #boxs_ [540,4] find ious with the highest confidence box(x_min,y_min,x_max,y_max)

    #transform parameter px,py,pw,ph
    px_min = boxes_default[:,4]
    py_min = boxes_default[:,5]
    px_max = boxes_default[:,6]
    py_max = boxes_default[:,7]
    px = (px_min+px_max)/2
    py = (py_min+py_max)/2
    pw = px_max-px_min
    ph = py_max-py_min
    #transform parameter px,py,pw,ph (end)
    #extract max,min position of boxs_
    tx = boxs_[:,0] #center x
    ty = boxs_[:,1]
    tw = boxs_[:,2] #center y
    th = boxs_[:,3]
    gx = pw*tx+px
    gy = py*ty+py
    gw = pw*np.exp(tw)
    gh = ph*np.exp(th)
    box_minx = gx-gw/2
    box_maxx = gx+gw/2
    box_miny = gy-gh/2
    box_maxy = gy+gh/2
    # extract the highest row
    highest_minx = box_minx[highest_idx]
    highest_maxx = box_maxx[highest_idx]
    highest_miny = box_miny[highest_idx]
    highest_maxy = box_maxy[highest_idx]
    inter = np.maximum(np.minimum(box_maxx,highest_maxx)-np.maximum(box_minx,highest_minx),0)*np.maximum(np.minimum(box_maxy,highest_maxy)-np.maximum(box_miny,highest_miny),0)
    area_a = (box_maxx-box_minx)*(box_maxy-box_miny)
    area_b = (highest_maxx-highest_minx)*(highest_maxy-highest_miny)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    N = len(boxs_default)
    for i in range(N):
        if ious_true[i]: #has an object
            ann_confidence[i,cat_id] = 1
            ann_confidence[i,3] = 0
            #boxdefault:px,py,pw,ph
            px_start = boxs_default[i,4]
            py_start = boxs_default[i,5]
            px_end = boxs_default[i,6]
            py_end = boxs_default[i,7]
            pw = px_end-px_start
            ph = py_end-py_start
            px = (px_start+px_end)/2
            py = (py_start+py_end)/2
            #ground truth
            gx = (x_min+x_max)/2
            gy = (y_min+y_max)/2
            gw = x_max-x_min
            gh = y_max-y_min
            #ann_box tranformed tx,ty,tw,th
            ann_box[i,0] = (gx-px)/pw
            ann_box[i,1] = (gy-py)/ph
            ann_box[i,2] = np.log(gw/pw)
            ann_box[i,3] = np.log(gh/ph)

    ious_true = np.argmax(ious)
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    if ious[ious_true]<threshold:
        ann_confidence[ious_true,cat_id] = 1
        ann_confidence[ious_true,-1] = 0
        #boxdefault:px,py,pw,ph
        px_start = boxs_default[i,4]
        py_start = boxs_default[i,5]
        px_end = boxs_default[i,6]
        py_end = boxs_default[i,7]
        pw = px_end-px_start
        ph = py_end-py_start
        px = (px_start+px_end)/2
        py = (py_start+py_end)/2
        #ground truth
        gx = (x_min+x_max)/2
        gy = (y_min+y_max)/2
        gw = x_max-x_min
        gh = y_max-y_min
        #ann_box tranformed tx,ty,tw,th
        ann_box[i,0] = (gx-px)/pw
        ann_box[i,1] = (gy-py)/ph
        ann_box[i,2] = np.log(gw/pw)
        ann_box[i,3] = np.log(gh/ph)
    return ann_box,ann_confidence


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
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
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        if self.anndir == "data/test/annotations/":
            pass
        else:
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
        if self.anndir == "data/test/annotations/":
            return image
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.

        anno_txt = open(ann_name)
        anno = anno_txt.readlines()
        anno_txt.close()
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        for i in range(len(anno)):
            line = anno[i].split()
            cat_id = int(line[0])
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
            ann_box,ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,cat_id,x_min,y_min,x_max,y_max)
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)            
        return image, ann_box, ann_confidence

class Aug(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        image = cv2.imread(img_name)
        x_shape0 = image.shape[1]
        y_shape0 = image.shape[0]
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        anno_txt = open(ann_name)
        anno = anno_txt.readlines()
        anno_txt.close()
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        crop_x_min,crop_y_min = x_shape0,y_shape0
        crop_x_max,crop_y_max = 0,0
        for i in range(len(anno)):
            line = anno[i].split()
            cat_id = int(line[0])
            x_start = float(line[1])
            y_start = float(line[2])
            w = float(line[3])
            h = float(line[4])
            x_end = x_start+w
            y_end = y_start+h
            # looking all the boxes, finding the boudary
            crop_x_min = min(x_start,crop_x_min)
            crop_x_max = max(x_end,crop_x_max)
            crop_y_min = min(y_start,crop_y_min)
            crop_y_max = max(y_end,crop_y_max)
        crop_x_start = random.uniform(0, int(crop_x_min))
        crop_y_start = random.uniform(0,int(crop_y_min))
        crop_x_end = random.uniform(int(crop_x_max),int(x_shape0))
        crop_y_end = random.uniform(int(crop_y_max),int(y_shape0))
        I = image[int(crop_y_start):int(crop_y_end),int(crop_x_start):int(crop_x_end),:]
        x_shape1 = I.shape[1]
        y_shape1 = I.shape[0]
        I = cv2.resize(I,(320,320))
        I = np.transpose(I,(2,0,1))
        for i in range(len(anno)):
            line = anno[i].split()
            cat_id = int(line[0])
            x_start = float(line[1])
            y_start = float(line[2])
            w = float(line[3])
            h = float(line[4])
            x_end = x_start+w-crop_x_start
            y_end = y_start+h-crop_y_start
            x_min = (x_start-crop_x_start)/x_shape1
            y_min = (y_start-crop_y_start)/y_shape1
            x_max = (x_end)/x_shape1
            y_max = (y_end)/y_shape1
            ann_box,ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,cat_id,x_min,y_min,x_max,y_max)     
        return I, ann_box, ann_confidence
