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




def YOLO_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from YOLO, [batch_size, 5,5, num_of_classes]
    #pred_box        -- the predicted bounding boxes from YOLO, [batch_size, 5,5, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, 5,5, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, 5,5, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for YOLO
    #
    #For confidence (class labels), use cross entropy (F.binary_cross_entropy)
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    pred_confidence = pred_confidence.reshape(-1,4)
    pred_box = ann_box.reshape(-1,4)
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
    L_yolo = L_cls+L_box
    return L_yolo
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*5*5, num_of_classes]
    #and reshape box to [batch_size*5*5, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.



    '''
    YOLO network
    Please refer to the hand-out for better visualization

    N is batch_size
    bn is batch normalization layer
    relu is ReLU activation layer
    conv(cin,cout,ksize,stride) is convolution layer
    cin - the number of input channels
    cout - the number of output channels
    ksize - kernel size
    stride - stride
    padding - you need to figure this out by yourself

    input -> [N,3,320,320]

    conv(  3, 64, 3, 2),bn,relu -> [N,64,160,160]

    conv( 64, 64, 3, 1),bn,relu -> [N,64,160,160]
    conv( 64, 64, 3, 1),bn,relu -> [N,64,160,160]
    conv( 64,128, 3, 2),bn,relu -> [N,128,80,80]

    conv(128,128, 3, 1),bn,relu -> [N,128,80,80]
    conv(128,128, 3, 1),bn,relu -> [N,128,80,80]
    conv(128,256, 3, 2),bn,relu -> [N,256,40,40]

    conv(256,256, 3, 1),bn,relu -> [N,256,40,40]
    conv(256,256, 3, 1),bn,relu -> [N,256,40,40]
    conv(256,512, 3, 2),bn,relu -> [N,512,20,20]

    conv(512,512, 3, 1),bn,relu -> [N,512,20,20]
    conv(512,512, 3, 1),bn,relu -> [N,512,20,20]
    conv(512,256, 3, 2),bn,relu -> [N,256,10,10]

    conv(256,256, 1, 1),bn,relu -> [N,256,10,10]
    conv(256,256, 3, 2),bn,relu -> [N,256,5,5] (the last hidden layer)

    output layer 1 - confidence
    (from the last hidden layer)
    conv(256,num_of_classes, 3, 1),softmax -> [N,num_of_classes,5,5]
    permute (or transpose) -> [N,5,5,num_of_classes]

    output layer 2 - bounding boxes
    (from the last hidden layer)
    conv(256, 4, 3, 1) -> [N,4,5,5]
    permute (or transpose) -> [N,5,5,4]
    '''


class YOLO(nn.Module):

    def __init__(self, class_num):
        super(YOLO, self).__init__()
        
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
        self.layer_bbox = nn.Sequential(
            nn.Conv2d(256,4,3,1,1)
        )
        self.layer_conf = nn.Sequential(
            nn.Conv2d(256,4,3,1,1),
            nn.Softmax()
        )

    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
    
        #TODO: define forward
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
        x = self.layer10(x)
        x = self.layer11(x)
        bboxes = self.layer_bbox(x)
        confidence = self.layer_conf(x)
        """ bboxes = np.transpose(bboxes,(1,2,0))
        confidence = np.transpose(bboxes,(1,2,0)) """
        bboxes = bboxes.permute(0,2,3,1)
        confidence = confidence.permute(0,2,3,1)
        #remember to apply softmax to confidence! Which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,5,5,num_of_classes]
        #bboxes - [batch_size,5,5,4]
        
        return confidence,bboxes










