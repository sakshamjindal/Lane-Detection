
import numpy as np
import math
import torch
from skimage.morphology import label


def get_metric(metric_name):
    if metric_name== "f_score":
        return fscore_batch
    else:
        raise NotImplementedError("This metric is not implemented yet")


def IOU(y_true_in,y_pred_in):
    
    labels = y_true_in
    outputs = y_pred_in

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    return iou

#     true_objects = len(np.unique(labels))
#     pred_objects = len(np.unique(y_pred))

#     intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

#     # Compute areas (needed for finding the union between all objects)
#     area_true = np.histogram(labels, bins = true_objects)[0]
#     area_pred = np.histogram(y_pred, bins = pred_objects)[0]
#     area_true = np.expand_dims(area_true, -1)
#     area_pred = np.expand_dims(area_pred, 0)

#     # Compute union
#     union = area_true + area_pred - intersection

#     # Exclude background from the analysis
#     intersection = intersection[1:,1:]
#     union = union[1:,1:]
#     union[union == 0] = 1e-9

#     SMOOTH = 1e-6
#     # Compute the intersection over union
#     iou = (intersection + SMOOTH) / (union + SMOOTH)
    
#     return iou

def pixel_accuracy(y_true_in,y_pred_in):
    
    correct = (y_true_in == y_pred_in)
    correct = np.sum(correct)
    
    return (correct*1.0)/(correct.shape[0]*correct.shape[1])

        
def fscore(y_true_in, y_pred_in,beta= 1, print_table = False):
    """As described here https://www.kaggle.com/c/airbus-ship-d etection#evaluation
    
    y_true_in: A numpy array of shape [H, W]. This is actual
    y_pred_in: A numpy array of shape [H, W]. This is predicted
    
    """
    #labels = label(y_true_in > threshold)
    #y_pred = label(y_pred_in > threshold)
    
    iou = IOU(y_true_in,y_pred_in)
    
    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    threshold_ = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    beta2 = math.pow(2, beta)
    
    for t in [0.1]:
        tp, fp, fn = precision_at(t, iou)
        if ((1+beta2)*tp) + fp + (beta2*fn) > 0:
            p = ((1+beta2)*tp) / ( ((1+beta2)*tp) + fp + (beta2*fn))
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
        threshold_.append(t)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    score = np.sum(prec)/len(threshold_)
    
    return score,iou


def fscore_batch(targets, outputs):
        
    SMOOTH = 1e-6 
    
    _ , pred = torch.max(outputs,dim=1)
    pred = pred.cpu().numpy()
    true = targets.cpu().numpy()
    
    outputs = pred
    labels = true
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
        
    return np.mean(iou)

def pixel_acc(pred, label):
	_, preds = torch.max(pred, dim=1)
	valid    = (label>=1).long()
	acc_sum = torch.sum(valid * (preds == label).long())
	pixel_sum = torch.sum(valid)
	acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
	return acc
   
# def IoU(outputs,targets):
    
#     SMOOTH = 1e-6
#     _, outputs = torch.max(outputs,dim=1)
#     labels = targets
            
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
        
#     return iou

# def fscore_batch(targets, outputs):
        
#     _ , pred = torch.max(outputs,dim=1)
#     pred = pred.cpu().numpy()
#     true = targets.cpu().numpy()
    
#     batch_size = true.shape[0]
#     fscore_m, iou_m, pixel_m  = [] , [] , []
#     for batch in range(batch_size):
#         #fscore_o,iou = fscore(true[batch], pred[batch])
#         iou = IOU(true[batch], pred[batch])
#         #pixel_accuracy = pixel_accuracy(true[batch], pred[batch])
#         print(iou)
#         #fscore_m.append(fscore)
#         iou_m.append(iou)
#         #pixel_m.append(pixel_accuracy)
        
#     return np.mean(iou_m)