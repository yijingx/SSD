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
batch_size = 16



boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True


if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    dataset_aug = Aug("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_aug = torch.utils.data.DataLoader(dataset_aug, batch_size=batch_size, shuffle=True, num_workers=0)

    
    
    # for i, data in enumerate(dataloader_aug, 0):
    #     images_, ann_box_, ann_confidence_ = data
    #     visualize_pred("test", ann_confidence_[0].numpy(), ann_box_[0].numpy(), ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(),boxs_default) 
    
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
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1
        
        for i, data in enumerate(dataloader_aug, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default,epoch,save=True)

        
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
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default,epoch,save=True)
        
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
    #dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, image_size=320)
    #dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()

    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320,wholedataset=True) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, image_size=320,wholedataset=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=0)
    #train dataset
    for i, data in enumerate(dataloader, 0):
        images_, ann_box_, ann_confidence_,x_shape,y_shape = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)
        for j in range(16):
            pred_confidence_ = pred_confidence[j].detach().cpu().numpy()
            pred_box_ = pred_box[j].detach().cpu().numpy()
            x_shape_ = int(x_shape[j])
            y_shape_ = int(y_shape[j])
            #visualize_pred("test",  pred_confidence_, pred_box_, pred_confidence_, pred_box_, images_[j].numpy(), boxs_default,threshold=0.3)

            pred_confidence_NMS,pred_box_NMS,ls_pos = non_maximum_suppression(pred_confidence_.copy(),pred_box_.copy(),boxs_default,threshold=0.5)
            

            # txt_id = str(int(i)*16+j)
            # s = txt_id.zfill(5)
            # ann_name = "predicted_boxes/train/"+s+".txt"
            # f = open(ann_name,"w")
            # for line in ls_pos:
            #     line_ = str(line[0])+" "+str(round(line[1]*x_shape_,2))+" "+str(round(line[2]*y_shape_,2))+" "+str(round(line[3]*x_shape_,2))+' '+str(round(line[4]*y_shape_,2))
            #     f.write(line_)
            #     f.write("\n")
            #     print(s,line_)
            # f.close()

            #TODO: save predicted bounding boxes and classes to a txt file.
            #you will need to submit those files for grading this assignment
            visualize_pred("test", pred_confidence_NMS, pred_box_NMS, pred_confidence_, pred_box_, images_[j].numpy(), boxs_default,threshold=0.5)
            #visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[j].numpy(), ann_box_[j].numpy(),images_[j].numpy(), boxs_default,threshold=0.5)
            cv2.waitKey(1000)
    ##test dataset
    # for i, data in enumerate(dataloader_test, 0):
    #     images_,x_shape,y_shape = data
    #     images = images_.cuda()
    #     #ann_box = ann_box_.cuda()
    #     #ann_confidence = ann_confidence_.cuda()

    #     pred_confidence, pred_box = network(images)
    #     for j in range(16):
    #         pred_confidence_ = pred_confidence[j].detach().cpu().numpy()
    #         pred_box_ = pred_box[j].detach().cpu().numpy()
    #         x_shape_ = int(x_shape[j])
    #         y_shape_ = int(y_shape[j])
    #         #visualize_pred("test",  pred_confidence_, pred_box_, pred_confidence_, pred_box_, images_[j].numpy(), boxs_default,threshold=0.3)

    #         pred_confidence_NMS,pred_box_NMS,ls_pos = non_maximum_suppression(pred_confidence_.copy(),pred_box_.copy(),boxs_default,threshold=0.5)
            

    #         txt_id = str(int(i)*16+j)
    #         s = txt_id.zfill(5)
    #         ann_name = "predicted_boxes/test/"+s+".txt"
    #         f = open(ann_name,"w")
    #         for line in ls_pos:
    #             line_ = str(line[0])+" "+str(round(line[1]*x_shape_,2))+" "+str(round(line[2]*y_shape_,2))+" "+str(round(line[3]*x_shape_,2))+' '+str(round(line[4]*y_shape_,2))
    #             f.write(line_)
    #             f.write("\n")
    #             print(s,line_)
    #         f.close()

    #         #TODO: save predicted bounding boxes and classes to a txt file.
    #         #you will need to submit those files for grading this assignment
    #         #visualize_pred("test", pred_confidence_NMS, pred_box_NMS, pred_confidence_, pred_box_, images_[j].numpy(), boxs_default,threshold=0.5)
    #         #cv2.waitKey(1000)

