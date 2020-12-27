# Obective

Create a solution to read house numbers.  

# Dataset

We are going to use The Street View House Numbers (SVHN) Dataset. Download the dataset and keep it in dataset folder. The dataset has 3 folders train, test, and extra. To create a bigger the dataset I have merged the train and extra folders into a combined folder, out of which we are training on 90% of the data and testing on the 10% of the data.

To check out the dataset, run this notebook --> https://github.com/trinanjan12/svhn_digit_recognizer/blob/master/dataset/explore_dataset.ipynb <br>
To create annotations from the give mat obj run this notebook --> https://github.com/trinanjan12/svhn_digit_recognizer/blob/master/dataset/explore_dataset.ipynb <br>

# Approaches
Since the dataset comes with bbox information we can run state of the art **object detection models like Detectron 2 library or yolov5 model**. But before starting with these rather heavy techniques I wanted to pose the problem as a direct **multilabel classification** problem.

    1. Multilabel Classification 
    2. Object Detection using Detectron 2
    3. Object Detection using Yolov5

# 1. Multilabel Classification

In a multilabel classification, we train a network to predict multiple classes. Like our images has multiple numbers in the same image, so we have to increase probability of each class if it is present otherwise push it to 0. So we could think of it as binary classification but for each class.

Checkout this notebook for multilabel classification network train and testing --> 
for training -->
https://github.com/trinanjan12/svhn_digit_recognizer/blob/master/multilabel_classification/multilabel_train.ipynb 
<br>
for testing --> https://github.com/trinanjan12/svhn_digit_recognizer/blob/master/multilabel_classification/multilabel_test.ipynb 
<br><br>
**I have experimented with bigger dataset and using pretrained imagenet True or False** . Increasing dataset increases the accuracy. Ideally I think we should not use pretrained imagenet as imagenet images are quite different 

**Since our test dataset has images with different orientations i have used randomrotation as one of the augmentation technique**

**Results**<br> 
after inferencing on the model trained on bigger dataset i could see that the probabilities are higher for the class if the label is present. Also randomrotation helped big time. <br>
But there are problems with this approach, sometimes we have number like **2255**. In this case we will only predict 2 and 5 but can't predict how many 2 or 5. **So this is a big problem**

**Future work**<br>
Instead of using multilabel classification with this we can first use a network to find out the bbox of each digits and then use classification network(similar like mnist). Although this is what in a way detectron/yolov5 will do for us. That's why I moved to object detection.

# 2. Object Detection using Detectron 2
I have used the opensource object detection lib from facebook research. It has rich documentation along with apis to make your life easy.<br>
**checkput the original repo** : https://github.com/facebookresearch/detectron2
<br>

**Summary**
1. To train with detectron2 I first created the dataset in coco format.
script for this could be found here --> https://github.com/trinanjan12/svhn_digit_recognizer/blob/master/dataset/create_coco_format.ipynb 
2. model used is **faster_rcnn_R_101_FPN_3x** trained for **50000 iterations** on the training dataset of SVHN images
3. Train and test notebook links --> https://github.com/trinanjan12/svhn_digit_recognizer/tree/master/detectorn2/pred_output
4. Prediction for the **final test output** could be found here --> https://github.com/trinanjan12/svhn_digit_recognizer/tree/master/detectorn2/

**Results**<br> 
The predictions are good for images with horizontal flip. Although images with vertical flips are not giving very good output.very low resolution images suffer from bad output. 

**Future work** <br>
I couldn't findtune anymore because i believe to solve the problem of vertical flip we need to do augmentation similar to this during training. The vertical flip is off in detectron default config file. To use the custom augmentation we need to create a dataloader and then apply augmentation. ref 

# 3. Object Detection using Yolov5
I have used a brilliant open source lib  for my implementation of yolov5. 
**checkput the original repo** : https://github.com/ultralytics/yolov5
<br>

**Summary**
1. To train with Yolov5 I first created the dataset in the desired format.
script for this could be found here -->one for creating from the train folder and one for the combined dataset(train+extra) <br>
https://github.com/trinanjan12/svhn_digit_recognizer/blob/master/yolov5/create_yolo_anno.py <br>
https://github.com/trinanjan12/svhn_digit_recognizer/blob/master/yolov5/create_yolo_anno_full.py <br>
2. model used is **yolov5m** and trained for 50 epochs on the training dataset of SVHN images
3. Train and test notebook links --> training and test scripts are given in the repo
4. Prediction for the **final test output** could be found here --> https://github.com/trinanjan12/svhn_digit_recognizer/tree/master/yolov5/pred_output

**Results**<br> 
The predictions are better compared to detectron2. 

**Future work** <br>
Need to figure out a way to detect very low res image
