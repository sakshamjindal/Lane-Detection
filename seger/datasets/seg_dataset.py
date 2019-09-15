import time
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import os
import scipy.misc as m
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from skimage import io, util
from skimage.color import rgb2gray
from torchvision.transforms import functional as F
import albumentations as albu

import albumentations as albu

def get_training_augmentation():
    
    return albu.Compose([

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Transpose(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.3, border_mode=0),
        #albu.RandomCrop(height=320, width=320, always_apply=True),
        
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.RandomContrast(p=0.5),
        albu.HueSaturationValue(p=0.5),
        
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.3,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.3,
        ),

    ])

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing():
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def read_semantic_mask_format(list_file):
    """ If the output is given in the image format 
    """
    fnames, fmasks = list(), list()
    with open(list_file) as f:
            lines = f.readlines()
            num_samples = len(lines)

    for line in lines:
        splited = line.strip().split()
        fnames.append(splited[0])
        fmasks.append(splited[1])
        
        
    return num_samples, fnames, fmasks


class LaneDataset(Dataset):
    """Lane Road dataset"""
    
    CLASSES = ['lane','unlabelled']
    

    def __init__(self, cfg, train=True,classes=None,augmentations=False,mode="train"):
        """
        Args:
            list_file (string): Path to the csv file with image paths
            train (bool): 'train', 'valid', or 'test'
            augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
            preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
        """
            
        CLASSES = ['lane','unlabelled']
        
        self.mode = mode
        self.train = train
        self.cfg = cfg
        self.augmentations = augmentations
        
        if self.train:
            self.list_file = self.cfg["data"]["train_data"]
        else:
            self.list_file = self.cfg["data"]["valid_data"]
            
        if self.augmentations:
            self.augmentations = get_training_augmentation()
                 
        self.num_samples,self.fnames,self.fmasks = read_semantic_mask_format(self.list_file)
        
        if classes is None:
            classes = CLASSES
        
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        #self.mean(np.array([73, 77, 77]))
        self.std = np.array([0.11774686, 0.0926435 , 0.08401473])

    def __len__(self):
        return (self.num_samples)

    def __getitem__(self, idx):
                      
        image_path = self.cfg["data"]["train_root"] + self.fnames[idx]
        mask_path = self.cfg["data"]["label_root"] + self.fmasks[idx]
        
        # read data
        img = io.imread(image_path)
        img = np.array(img, dtype=np.uint8)

        lbl = io.imread(mask_path)
        lbl = rgb2gray(lbl)*255
        lbl = np.array(lbl, dtype=np.int32)
        
        # extract certain class from mask 
        mask = (lbl.copy()>127).astype('uint8')
           
        #apply augmentations
        data = {"image": img, "mask": mask}
        if self.augmentations:
            #print("Applying augmentation")
            sample = self.augmentations(**data)
            img, mask = sample["image"], sample["mask"]
            
        # apply transformations
        img, mask = self.transform(img,mask)
            
        return img, mask
    
    def transform(self,img,mask):
        
        if self.mode=="train":
            width, height = self.cfg["train"]["width"], self.cfg["train"]["height"]
        else:
            width, height= img.shape[1], img.shape[0]
        
        img = np.array(Image.fromarray(img).resize((width,height),Image.BILINEAR))
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        
        mask = np.array(Image.fromarray(mask*255).resize((width,height),Image.NEAREST))        
        mask = mask.astype(float)/255.0
        mask = torch.from_numpy(mask).long()
        
        return img, mask