import os
import random
import numpy as np

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




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.binary_cross_entropy)
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    pred_confidence = pred_confidence.reshape(-1,4)
    pred_box = pred_box.reshape(-1,4)
    ann_confidence = ann_confidence.reshape(-1,4)
    ann_box = ann_box.reshape(-1,4)
    N = pred_confidence.shape[0]
    idx_obj = []
    idx_empty = []
    for i in range(N):
        if ann_confidence[i,3]==1:
            idx_empty.append(i)
        else:
            idx_obj.append(i)
    L_cls = F.binary_cross_entropy(pred_confidence[idx_obj],ann_confidence[idx_obj])+3*F.binary_cross_entropy(pred_confidence[idx_empty],ann_confidence[idx_empty])
    L_box = F.smooth_l1_loss(pred_box[idx_obj],ann_box[idx_obj])
    L_ssd = L_cls+L_box
    return L_ssd




class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,3,2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,3,2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128,256,3,2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256,512,3,2,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(512,256,3,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(256,256,3,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer_a1 = nn.Sequential(
            nn.Conv2d(256,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer_a2 = nn.Sequential(
            nn.Conv2d(256,256,3,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer_a3 = nn.Sequential(
            nn.Conv2d(256,256,3,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(256,16,1,1,0)
        self.conv2 = nn.Conv2d(256,16,3,1,1)
        self.softm = nn.Softmax(2)

        
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        
        #remember to apply softmax to confidence! Which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer8(x)
        x = self.layer9(x)
        # divide into different size
        #100
        x_100 = x
        x_100_box = self.conv2(x_100)
        x_100_box = x_100_box.reshape([x_100_box.shape[0],16,100])
        x_100_ann = self.conv2(x_100)
        x_100_ann = x_100_ann.reshape([x_100_ann.shape[0],16,100])
        #25
        x_25 = self.layer_a1(x_100)
        x_25 = self.layer_a2(x_25)
        x_25_box = self.conv2(x_25)
        x_25_box = x_25_box.reshape([x_25_box.shape[0],16,25])
        x_25_ann = self.conv2(x_25)
        x_25_ann = x_25_ann.reshape([x_25_ann.shape[0],16,25])
        #9
        x_9 = self.layer_a1(x_25)
        x_9 = self.layer_a3(x_9)
        x_9_box = self.conv2(x_9)
        x_9_box = x_9_box.reshape([x_9_box.shape[0],16,9])
        x_9_ann = self.conv2(x_9)
        x_9_ann = x_9_ann.reshape([x_9_ann.shape[0],16,9])
        #1
        x_1 = self.layer_a1(x_9)
        x_1 = self.layer_a3(x_1)
        x_1_box = self.conv1(x_1)
        x_1_box = x_1_box.reshape([x_1_box.shape[0],16,1])
        x_1_ann = self.conv1(x_1)
        x_1_ann = x_1_ann.reshape([x_1_box.shape[0],16,1])
        #concatenate
        bboxes = torch.cat((x_100_box,x_25_box,x_9_box,x_1_box),2)
        bboxes = bboxes.permute((0,2,1))
        bboxes = bboxes.reshape((bboxes.shape[0],540,4))
        confidence = torch.cat((x_100_ann,x_25_ann,x_9_ann,x_1_ann),2)
        confidence = confidence.permute((0,2,1))
        confidence = confidence.reshape((confidence.shape[0],540,4))
        confidence = self.softm(confidence)
        return confidence,bboxes










