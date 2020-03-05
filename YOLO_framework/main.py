import argparse
import os
import numpy as np
import time
import cv2

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

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 100
batch_size = 32


#Create network
#fake YOLO 5x5
network = YOLO(class_num)
network.cuda()
cudnn.benchmark = True


if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, train = True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, train = False, image_size=320)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

    #test visualize
    for i, data in enumerate(dataloader, 0):
        images_, ann_box_, ann_confidence_ = data
        #visualize_pred("test", ann_confidence_[0].numpy(), ann_box_[0].numpy(), ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy())
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()

    for epoch in range(num_epochs):
        #TRAIN
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = YOLO_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy())
        
        
        #TEST
        network.eval()
        
        # TODO: split the dataset into 80% training and 20% testing
        # use the training set to train and the testing set to evaluate
        
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), precision_,recall_,thres)
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy())
        
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        
        
        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth')


else:
    #TEST
    dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, train = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_ = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        #pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files produced by SSD for grading. better practice it now.
        
        visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy())
        cv2.waitKey(1000)


