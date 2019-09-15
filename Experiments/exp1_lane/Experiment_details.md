# **Experiment Details**

### **Data Details:**

Total Number of Training Images : 450 <br />
Total Number of Validation Images : 450 <br />
Batch Size : 32

### **Training Details:**

**Number of Epochs** : 200 <br />
**Image Size** :  (224,224) <br />
**Number of classes** : 2 [background class considered as one class with index 0] <br />

**Optimizer Used** : Adam with base lr 0.01 <br />
**Scheduler Used** : SGDR with cossine annealing(Warm restart after 10,30,70,150 .. epochs) with max_lr =0.01 and 0.0001 <br />
**Data Parallel** : No <br />
**Loss Criterion** : Weighed Cross-entropy loss (Weights given- 1:200) <br />
**Network Architecture** : Deeplabv3 with Resnet 101 backbone <br />
**Pretrained** :  Encoder pretrained on Imagenet <br />
**Train-time augmentations** : Random Crop, Random Horizontal Flip, Random Vertical Flip,Random Transpose, Random Brightness, Random ShiftScaleRotate,Random Contrast, Random Perspective Transform, Random Hue Saturation Value, Normalization

### **Evaluation Criteria**

Mean IOU (Jaccard Index)

### **Experiment Result


### **Mask Visualisation** 

 
<!-- **Training Configuration**

| Parameters 	| Values	|
|----------	|:-------------	|
| Number of Epochs 	    | 200                                                       	|
| Image Size 	        | (224,224)  	                                                | 
| Number of classes 	|  2 [background class considered as one class with index 0] 	|
| Optimizer Used 	|  Adam with base lr	|
| Scheduler Used	|   SGDR with cossine annealing(Warm restart after 10,30,70,150 .. epochs) with max_lr =0.01 and 0.0001	|
| Data Parallel 	| No |
| Loss Criterion 	| Weighed Cross-entropy loss (Weights given- 1:200) |
| Network Architecture | Encoder pretrained on Imagenet |
| Augmentations | Random Crop, Random Horizontal Flip, Random Vertical Flip, Random Transpose, Random Brightness, Random ShiftScaleRotate, Random Contrast -->

<!-- Train-time augmentations : 
- Random Crop
- Random Horizontal Flip
- Random Vertical Flip
- Random Transpose
- Random Brightness
- Random ShiftScaleRotate
- Random Contrast -->










