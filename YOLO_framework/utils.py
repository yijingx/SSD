import numpy as np
import cv2


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use red green blue to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from YOLO, [5,5, num_of_classes]
    #pred_box        -- the predicted bounding boxes from YOLO, [5,5,4]
    #ann_confidence  -- the ground truth class labels, [5,5, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [5,5, 4]
    #image_          -- the input image to the network
    
    size,_,class_num = pred_confidence.shape
    #size = 5, the size of the output grid
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
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for yind in range(size):
        for xind in range(size):
            for j in range(class_num):
                if ann_confidence[yind,xind,j]>0.5: #if the network/ground_truth has high confidence on cell[yind,xind] with class[j]
                    #TODO:
                    #image1: draw ground truth bounding boxes on image1
                    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                    #parameters from ann_box
                    ann_rele_x = ann_box[yind,xind,0] #center,reletive
                    ann_rele_y = ann_box[yind,xind,1]
                    ann_width = ann_box[yind,xind,2]**2 #sqrt
                    ann_height = ann_box[yind,xind,3]**2
                    #calculate the relative position of start and end points
                    ann_rele_center_x = (ann_rele_x + xind)/size
                    ann_rele_center_y = (ann_rele_y + yind)/size
                    ann_rele_start_x = ann_rele_center_x-ann_width/2
                    ann_rele_start_y = ann_rele_center_y-ann_height/2
                    ann_rele_end_x = ann_rele_center_y+ann_width/2
                    ann_rele_end_y = ann_rele_center_y+ann_height/2
                    #you can use cv2.rectangle as follows:
                    start_point1 = (int(ann_rele_start_x*image1.shape[1]), int(ann_rele_start_y*image1.shape[0])) #top left corner, x1<x2, y1<y2
                    end_point1 = (int(ann_rele_end_x*image1.shape[1]), int(ann_rele_end_y*image1.shape[0])) #bottom right corner
                    color = colors[j] #use red green blue to represent different classes
                    thickness = 2
                    cv2.rectangle(image1, start_point1, end_point1, color, thickness)
                    #image2
                    start_point2 = (int(xind/size*image1.shape[1]),int(yind/size*image1.shape[0]))
                    end_point2 = (int((xind+1)/size*image1.shape[1]),int((yind+1)/size*image1.shape[0]))
                    cv2.rectangle(image2, start_point2, end_point2, color, thickness)
    
    #pred
    for yind in range(size):
        for xind in range(size):
            for j in range(class_num):
                if pred_confidence[yind,xind,j]>0.5:
                    #TODO:
                    #image3: draw network-predicted bounding boxes on image3
                    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                    #parameters from ann_box_box
                    pred_rele_x = pred_box[yind,xind,0] #center,reletive
                    pred_rele_y = pred_box[yind,xind,1]
                    pred_width = pred_box[yind,xind,2]**2 #sqrt
                    pred_height = pred_box[yind,xind,3]**2
                    #calculate the relative position of start and end points
                    pred_rele_center_x = (pred_rele_x + xind)/size
                    pred_rele_center_y = (pred_rele_y + yind)/size
                    pred_rele_start_x = pred_rele_center_x-pred_width/2
                    pred_rele_start_y = pred_rele_center_y-pred_height/2
                    pred_rele_end_x = pred_rele_center_y+pred_width/2
                    pred_rele_end_y = pred_rele_center_y+pred_height/2
                    #you can use cv2.rectangle as follows:
                    start_point1 = (int(pred_rele_start_x*image1.shape[1]), int(pred_rele_start_y*image1.shape[0])) #top left corner, x1<x2, y1<y2
                    end_point1 = (int(pred_rele_end_x*image1.shape[1]), int(pred_rele_end_y*image1.shape[0])) #bottom right corner
                    color = colors[j] #use red green blue to represent different classes
                    thickness = 2
                    cv2.rectangle(image1, start_point1, end_point1, color, thickness)
                    #image2
                    start_point2 = (int(xind/size*image1.shape[1]),int(yind/size*image1.shape[0]))
                    end_point2 = (int((xind+1)/size*image1.shape[1]),int((yind+1)/size*image1.shape[0]))
                    cv2.rectangle(image2, start_point2, end_point2, color, thickness)
    
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


#this is an example implementation of IOU.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxes, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 4], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_min,y1_min,x1_max,y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxes[:,2],x_max)-np.maximum(boxes[:,0],x_min),0)*np.maximum(np.minimum(boxes[:,3],y_max)-np.maximum(boxes[:,1],y_min),0)
    area_a = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)


#this function will be used later
'''
def non_maximum_suppression(confidence_t, box_t, overlap=0.5, threshold=0.5):
    #input:
    #confidence_t -- the predicted class labels from YOLO, [5,5, num_of_classes]
    #box_t        -- the predicted bounding boxes from YOLO, [5,5, 4]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    size,_,class_num = confidence_t.shape
    #size = 5, the size of the output grid
    #class_num = 4
    
    #TODO: non maximum suppression
    #you can reshape the confidence and box to [5*5,class_num], [5*5,4], for better indexing
'''















