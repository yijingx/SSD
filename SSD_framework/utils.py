import numpy as np
import cv2
from dataset import iou


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use red green blue to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default,epoch = 500,save=False):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]          
    image4[:]=image[:]       
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                px_start = boxs_default[i,4]
                py_start = boxs_default[i,5]
                px_end = boxs_default[i,6]
                py_end = boxs_default[i,7]
                pw = px_end-px_start
                ph = py_end-py_start
                px = (px_start+px_end)/2
                py = (py_start+py_end)/2
                # ann_box tranformed tx,ty,tw,th
                dx = ann_box[i,0]
                dy = ann_box[i,1]
                dw = ann_box[i,2]
                dh = ann_box[i,3]
                # transformed again
                gx = pw*dx+px
                gy = ph*dy+py   
                gw = pw*np.exp(dw)
                gh = ph*np.exp(dh)
                ann_rele_start_x = gx-gw/2
                ann_rele_start_y = gy-gh/2
                ann_rele_end_x = gx+gw/2
                ann_rele_end_y = gy+gh/2            
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                start_point1 = (int(ann_rele_start_x*image1.shape[1]), int(ann_rele_start_y*image1.shape[0])) #top left corner, x1<x2, y1<y2
                end_point1 = (int(ann_rele_end_x*image1.shape[1]), int(ann_rele_end_y*image1.shape[0])) #bottom right corner
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                cv2.rectangle(image1, start_point1, end_point1, color, thickness)
                #image2
                start_point2 = (int(px_start*image1.shape[1]),int(py_start*image1.shape[0]))
                end_point2 = (int(px_end*image1.shape[1]),int(py_end*image1.shape[0]))
                cv2.rectangle(image2, start_point2, end_point2, color, thickness)
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                px_start = boxs_default[i,4]
                py_start = boxs_default[i,5]
                px_end = boxs_default[i,6]
                py_end = boxs_default[i,7]
                pw = px_end-px_start
                ph = py_end-py_start
                px = (px_start+px_end)/2
                py = (py_start+py_end)/2
                #
                dx_pred = pred_box[i,0]
                dy_pred = pred_box[i,1]
                dw_pred = pred_box[i,2]
                dh_pred = pred_box[i,3]
                #
                gx_= pw*dx_pred+px
                gy = ph*dy_pred+py
                gw = pw*np.exp(dw_pred)
                gh = ph*np.exp(dh_pred)
                #
                ann_rele_start_x = gx-gw/2
                ann_rele_start_y = gy-gh/2
                ann_rele_end_x = gx+gw/2
                ann_rele_end_y = gy+gh/2
                start_point3 = (int(ann_rele_start_x*image1.shape[1]), int(ann_rele_start_y*image1.shape[0])) #top left corner, x1<x2, y1<y2
                end_point3 = (int(ann_rele_end_x*image1.shape[1]), int(ann_rele_end_y*image1.shape[0])) #bottom right corner
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                cv2.rectangle(image3, start_point3, end_point3, color, thickness)
                start_point4 = (int(px_start*image1.shape[1]),int(py_start*image1.shape[0]))
                end_point4 = (int(px_end*image1.shape[1]),int(py_end*image1.shape[0]))
                cv2.rectangle(image4, start_point4, end_point4, color, thickness)


    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.
    if save==True:
        cv2.imwrite("./visual_res/%s_%d.jpg"%(windowname,epoch),image)




def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    l_box = []
    N = len(confidence_)
    highest_conf, highest_class = torch.max(confidence_[:,0:2],1) #[num_of_boxes,1],[num_of_boxes,1]
    highest_conf_ofall, highest_idx = torch.max(highest_conf,0) #1,1
    while highest_conf_ofall>threshold:
        confidence_[highest_idx] = [0,0,0,0]
        #add to new list(box)
        l_box.append(box_[highest_idx])
        for i in range(N):
            x_min = box_[i,0]
            ious =  iou(boxs_default, box_[i,0],box_[i,1],box_[i,2],box_[i,3])
            l_idx = (ious>overlap).squeeze().nonzero()
            confidence_[l_idx] = [0,0,0,0] #remove all the overlap box from confidence_
        highest_conf, highest_class = torch.max(confidence_[:,0:2],1) #[num_of_boxes,1],[num_of_boxes,1]
        highest_conf_ofall, highest_idx = torch.max(highest_conf,0) #1,1
    return l_box

    #TODO: non maximum suppression
    












