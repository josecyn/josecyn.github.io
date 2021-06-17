---
layout: post
title:  "Steel Beam Counting (part 1)"
description: "Counting of steel beam objects using segmentation."
type: card-dated
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
categories: post
tags: 
  - Pytorch
  - Computer Vision
  - Segmentation
author: Jose
card: card-2
---

# Overview

In this project, the task is to count steel beam objects in the image. I was asked to do it using segmentation models.
At first, I tried using the well-known [Mask RCNN repository](https://github.com/matterport/Mask_RCNN) but the results were not quite good.

After unsuccessfully trying the Mask R-CNN model, I decided to try the __UNet architecture__.
The code is based on the repository [Robot surgery segmentation](https://github.com/ternaus/robot-surgery-segmentation).

This architecture has been successful in many Kaggle segmentation competitions such as [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018#description) or [Carvana](https://www.kaggle.com/c/carvana-image-masking-challenge).

I took many ideas from the description of the [winning solution](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741) of the Data Science Bowl competition.

The base idea is to use an __image-to-image UNet architecture__ using a deep encoder preinitialized such as VGG-19 or ResNet152 and afterwards apply postprocessing to separate and count the objects in the image.

Objects in picture are very close one to another and it becomes very hard to separate them. For that reason, I decided to create two models: one will tell me all the __objects__ in the image, and the second one will tell me the __borders__ of those objects. In the postprocessing phse, I will have to __substract__ the boders from the objects. That will help me to count how many objects appear in the picture. For details why I needed these two models, check the section below [Step 5](#Step-5) under [Postprocessing](#Postprocessing).

# Dataset

In this task we have a small dataset with around 375 pictures that contains many different shapes. For some shapes we have many examples (20+) and for others we just have 1. In addition, we want our model to correctly predict unseen shapes.

The pictures have been taken in few different locations and conditions. This is perhaps the most challenging aspect.

After working on the Mask R-CNN model, I had around 295 labelled images. Images were labelled using [labelme](https://github.com/wkentaro/labelme).

## Problem And Idea

To tell apart one shape of an object from another when using segmentation it's a very hard task because objects are located very near of each other.
As a solution, I decided to try to have 2 different classes: __Object body__ and __object border__. 

In the post-processing section I would just consider the body of the object, and that would be a few pixels away from the next object hopefully.

### Objects Body Dataset

That was easy, because the creator of the labelling tool already provides a script that can handle that.
That script creates a VOC-like folder structure. I took the __target masks__ from the folder __SegmentationClassPNG__.

Example:

<img src="/assets/img/posts/beam_counting_p1/body_object.PNG" />

### Borders Dataset

That was a bit harder to build. First of all, I created a script that takes a json file with all the polygons and its points and saves every shape individually. 

After having each shape saved individually, I ran a script that takes every polygon, finds out its border and saves it in an array. When all polygons have been processed, the script adds all the borders to the same output mask. Finally, the output mask is stored.

For finding the border of the initial polygons I used OpenCV's `threshold`, `dilate` and `substract` functions.

Result:

<img src="/assets/img/posts/beam_counting_p1/border_object.PNG" />

## Dataset size

During the development of the Mask R-CNN I saw that there was a big difference between the shapes in the pictures. Some of them are round and thick while others are very thin. I realised that the dataset had way more thick and round examples than thin one. So I decided to create a more __balanced dataset based on the thickness__ of the objects. Because the thin objects are rarer, I ended up having a small dataset of only 80 training + 10 validation examples. Having so few examples, image augmentation became important. See details below.

After training with the models with both datasets, the models trained on the small dataset performed __slightly better__ than the others. So __I mostly used the smaller dataset__.

Based on the robot surgery segmentation implementation, I implemented my own dataset class.

# Model architecture

The [UNET](https://arxiv.org/abs/1505.04597) was developed by Olaf Ronneberger et al. for Bio Medical Image Segmentation. The architecture contains two paths. First path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a traditional stack of convolutional and max pooling layers. The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions. Thus it is an end-to-end fully convolutional network (FCN), i.e. it only contains Convolutional layers and does not contain any Dense layer because of which it can accept image of any size.

The original paper shows the architecture of the UNet like this:

<img src="/assets/img/posts/beam_counting_p1/UNet_arch.png" width="480" height="411"/>

I tried out different pretrained VGG models: VGG-11, VGG-16, VGG-19 and VGG-19 with Batch normalization and some pretrained ResNets: ResNet-34, ResNet-101 and ResNet-152. I got the best result with __ResNet-152__.

Both models, body and border, are based on the same architecture: __UNet with ResNet-152 pretrained on ImageNet dataset as encoder__.

To try some of these models I had to adapt the original code from TernausNet. Specifically I modified the `models.py` file to add support for VGG-19, VGG-19bn, ResNet-101 and ResNet-152.

# Training

I tried training the models resizing the images between 800 and 1280 pixels and batches of 1 or 2 images. I used [__Jaccard index__ (Intersection Over Union)](https://en.wikipedia.org/wiki/Jaccard_index) and [dice coefficient](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient) as the evaluation metric. I also tried AP but I didn't used it.

After 2000 steps the models began to be already good. Best results were found after 8000 steps.

## Image augmentations

I didn't have a lot of examples and I also wanted the model to perform well on unseen data, so I used __image augmentations during training__ a lot.

I changed from the library 'Albumentations' (used in the original respository) to [__imgaug__](https://imgaug.readthedocs.io/en/latest/index.html), which I was more familiar with.

The image augmentations I used were:

- Random horizontal and vertical flips.
- Random color augmentation: contrast, hue and saturation.
- Random rotation between -350 and 350 degrees.
- Random crop and pad between -100 and 250 pixels.
- Random changes in the texture: MotionBlur, GaussianBlur, Dropout, etc.
- Random POV change: shear and perspective transform.

I also tried taking only small random crops from the image, but it didn't work as expected.

# Postprocessing

The postprocessing task using UNet models is more complex than using Mask R-CNN.

The basic steps of the whole process are the following:

1. __Predict__ the body and border mask and apply a threshold 140.
2. __Increase the body area__ a bit (opencv 'dilate').
3. __Decrease the border area__ a bit (opencv 'erode').
4. __Substract__ the border mask to the body mask.
5. __Find the contour areas__ using opencv's 'findContours' method.
6. __Filter actual object__ areas from previous contours.

Now, I'll explain each step more carefully.

## Steps 1 - 4

I use the models to predict body objects and border objects for the same image.
The pixel values range from 0 to 149. So, I remove the pixels in which the model was not "very sure", that means applying a threshold of 140 and removing pixels with lower values than 140.
At the end of this first stage, I increase a bit the area of the body prediction to fill gaps (using opencv dilate function) and I reduce the border area (using opencv erode function). I reduce the area of the border mask, because the model was trained on a training set that had thick borders.

As an example, take a look at this collage:

<img src="/assets/img/posts/beam_counting_p1/collage_2.jpg" width="750" height="250"/>

On the left, the 1st picture, there is the body prediction. Pay attention and look the gap between objects.
On the middle, the 2nd picture, there is the border prediction.
On the right, the 3rd picture, there is the result of the steps 1 to 4. As you can see, __the gaps between objects are much clear__ than in the picture number 1. That will help a lot the following steps of the postprocessing.

## Step 5

At this moment, the result of the steps 1 to 4 is going to be processed to find out the areas on the image.

To find out such areas, I use the method ['findContours'](https://docs.opencv.org/3.4.2/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a) from opencv.

The method returns a list of __contours__. Given a contour, using other opencv methods, you can get some properties such as area, bounding box, etc.

After I get the contours, I remove those which have an __area less than 200__. That filters dots and small lines so it speeds up the process.

It is __extremely important__ that the objects are isolated and don't touch each other. Here is an example when the objects are well separated __after postprocessing__:

<img src="/assets/img/posts/beam_counting_p1/opencv_4.PNG" width="452" height="422"/>
<img src="/assets/img/posts/beam_counting_p1/opencv_1.png" width="452" height="422"/>

On the first picture, the body prediction after postprocessing of steps 1 - 4. On the second picture, the result of 'findContours' method. Each different color represents a different contour. In that case, we have 4 objects + background.

Otherwise, if objects are not well isolated from each other, the result is that:

<img src="/assets/img/posts/beam_counting_p1/opencv_3.PNG" width="452" height="422"/>
<img src="/assets/img/posts/beam_counting_p1/opencv_2.PNG" width="452" height="422"/>

On the first picture, the body prediction without any processing. Because objects are not well delimited, 'findContours' method just finds __one single object__!

## Step 6

In the final step, once we have all the contours of the image, __it is time to count them__.

Because the predictions are imperfect, we end up with lines, dots and other areas that do not represent any object. In order to get rid of them, I had two approaches:

- __Simple area size arithmetic filtering__
- __Machine learning filtering__

### Area-based Arithmetic Filtering

In this simple approach, I take the biggest contour area of the image and set a limit in the X% of its area size. __All objects below this limit are not considered__. The reason of this approach is that objects usually are about the same size. With this limit we get rid of small objects.

After testing different limits, __30%__ had the best result.

Even though is a very simple approach, it was the __approach with better results__, even better than the machine learning!

Incovenients of this approach are that shapes of very different sizes in the same picture, will probably not be well classified. Another problem is when two objects are not well delimited; the resulting object will have a large area thus afecting the limit to be also very large and misclassifying smaller objects.

### Machine Learning Filtering

This approach was based on the winning Kaggle solution above. __I used a LightGBM model to predict IoU__.

I created one LightGBM model with few input parameters and another one with much more parameters. Here are the different parameters I used in training:

Ext1:
- __'left', 'top', 'width', 'height':__ bounding box that contains the contour
- __'bbox_area':__ area of the bounding box
- __'pol_area':__ area of the contour
- __'rrect_x', 'rrect_y', 'rrect_h', 'rrect_w':__ points of the 'reduced' bounding rectangle
- __'rrect_angle':__ angle of the reduced bounding rectangle
- __'rrect_area':__ area of the reduced bounding rectange
- __'ecircle_x', 'ecircle_y', 'ecircle_rad':__ points and radius of the bouding circle
- __'perimeter':__ perimeter of the contour
- __'max_area':__ size of the biggest area of all contours in picture
- __'min_area':__ size of the smallest area of all contours in picture
- __'n_contours':__ number of contours in picture
- __'metric':__ target IoU

Ext2:
- __'left', 'top', 'width', 'height':__ points of the bounding box
- __'pol_area':__ area of the bounding box
- __'max_area':__ size of the biggest area of all contours in picture
- __'min_area':__ size of the smallest area of all contours in picture
- __'metric':__ target IoU


First, I created a training csv dataset. The resulting csv contains one column for each parameter above.

To my surprise, __the simpler model (ext2) performed better than the more complex one (ext1)__. But both were not as good as the simple area size filtering explained above.

## End Result Evaluation

For the evaluation I created a script that ends up having a csv file that contains two columns: image name and number of objects.
So now is very simple to compare the output of the whole process against the test csv file to output the __accuracy__, __mse__ and number of __correct answers__.

# Result

The best model was composed by the following elements:

- Body model: __UNet with ResNet-152 encoder__ trained for around 8000 steps on a small dataset.
- Border model: __UNet with ResNet-152 encoder__ trained for around 20000 steps on a small dataset with thick borders.
- The filtering was made by __simple area size arithmetic filtering__ with limit set at 30%.

The final result is:

- __Accuracy: 0.7576__
- __MSE: 4.761__
- __Correct answers: 50__ out of 66 examples

Obviously, with such a small test dataset, this accuracy may change substancially.

## Some Correct Predictions

<img src="/assets/img/posts/beam_counting_p1/DSCF3568_collage.png" />
<img src="/assets/img/posts/beam_counting_p1/DSCF3691_collage.png" />
<img src="/assets/img/posts/beam_counting_p1/DSCF4267_collage.png" />

## Some Wrong Predictions

<img src="/assets/img/posts/beam_counting_p1/A3T0 (1)_collage.png" />
<img src="/assets/img/posts/beam_counting_p1/DSCF4558_collage.png" />

## Links

- [Robot surgery segmentation](https://github.com/ternaus/robot-surgery-segmentation)
- [TernausNet](https://github.com/ternaus/TernausNet)
- [Winning solution Kaggle Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741)
- [labelme labelling tool](https://github.com/wkentaro/labelme)
- [UNet paper](https://arxiv.org/abs/1505.04597)
- [Jaccard index (IoU)](https://en.wikipedia.org/wiki/Jaccard_index)
- [Dice coefficient](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient)
- [imgaug documentation](https://imgaug.readthedocs.io/en/latest/index.html)
- [imgaug github repository](https://github.com/aleju/imgaug)
- [laplotter](https://github.com/aleju/LossAccPlotter)
- [Pytorch's pretrained models](https://pytorch.org/docs/stable/torchvision/models.html)
- [Homography transformation 日本語 ](https://www.cellstat.net/homography/)
- [Opencv watershed 日本語](https://qiita.com/ysdyt/items/5972c9520acf6a094d90)
- [画像フィルタ処理](http://imagingsolution.blog.fc2.com/blog-entry-101.html)
- [Official Opencv watershed tutorial](https://docs.opencv.org/3.1.0/d3/db4/tutorial_py_watershed.html)
- [Opencv findContours](https://docs.opencv.org/3.4.2/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a)
- [Opencv morphological transformations](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
- [Opencv contour features tutorial](https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html)
- [Opencv thresholding](https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html)


