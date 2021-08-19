---
layout: post
permalink: /01ead88355127f33e70117da2578556e80ac1dbc/:categories/:year/:month/:day/:title:output_ext
title:  "Steel Beam Counting (part 2)"
description: "Counting of steel beam objects using object detection."
type: card-dated
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
categories: post
tags: 
  - Pytorch
  - Computer Vision
  - Object Detection
author: Jose
card: card-2
---

# Introduction

In part 1 of this project, following the requirements, I tried to count the object in the picture using segmentation.

I thought that this was not the best way so I proposed to apply object detection models to this problem.

After brief research, I decided to apply a novel promising model at the time, [CenterNet](https://github.com/xingyizhou/CenterNet), and a well-known model that I already used in previous projects [YOLOv3](https://github.com/AlexeyAB/darknet).

The development of the models was made in parallel, but here I will describe first the YOLOv3 process and then the CenterNet process.

The baseline to beat was **75.8% accuracy** from the segmentation-based solution.

## YOLOv3

I got a YOLOv3 model from a colleague that worked on that problem one year later. That model had **68.7% accuracy**.

After analyzing the data, I came up with 6 classes and different data distribution. I won't explain the details, but looking at the images I could see some 
patterns in the shapes of the different objects.

I already had experience with YOLOv3. I just needed to convert the labels to YOLOv3 format, tune the CFG file and run training.

**The first model I created had 75.4% accuracy**, almost matching the baseline achieved with the segmentation model and well above the previous YOLOv3 accuracy. 
And that was only the initial model, there was room for improvement.

At this point, more data sent from the customer bumped the number of images from 914 to 1627. 
On top of that, I focused my efforts on adjusting the values of the NMS algorithm. 
The objects in the images have very different shapes and after analyzing the output from the model,
I could spot some patterns that could be corrected by tuning the NMS function.

So, after applying the new data and tune the NMS parameters, **the accuracy of this second model increased from 75.4% to 83.6%**.

Lastly, I created a new model basically increasing the input resolution from 512px to 608px, and **the accuracy increased to 86.15%**.

I still thought that there was room for improvement in the NMS step, so I decided to use a different NMS implementation: Soft-NMS.
With this new implementation and some tunning, **the final accuracy achieved by the YOLOv3 model was: 89.74%.**

## CenterNet

After reading about CenterNet and seeing its results in "standard" datasets such as COCO or Imagenet I decided to apply it to this task.

For that, I used [xingyizhou's repository](https://github.com/xingyizhou/CenterNet). I had to modify the code to adapt it to this project's needs.

After a few tests, the best baseline was a `DLA-34` backbone model with **78.46% accuracy** already slightly better than the YOLOv3 model.

The next steps were to tune some parameters which gave me **80% accuracy**. 

After that, I examined the code and find out the possibility to apply `flip` at the time of inference. During this process, the inference takes place 
using the original image and the flipped version of the same image. The result with the higher confidence is the one taken. 
**With this technique the accuracy increased to 85.6%**.

Lastly, analog to the YOLOv3 process, I increased the input resolution from 512px to 608px.
With that change, **the final accuracy achieved by the CenterNet model was 93.85%**.

## More Test Data Available

After some time, I got more data from the customer: an additional batch of 537 pictures. 

I tested it with the CenterNet model **achieving 93.67% accuracy**, even though there were new kinds of objects and some 
pictures were taken in different environments.

No additional training was made by me with this new data.

## Conclusions

Here you have a chart with the evolution of the accuracy of both models:

![Graph comparison](/assets/img/posts/beam_counting_p2/steel_graph.png)

On the Y-axis is the accuracy and on the X-axis there are the different models I developed.

We can see that since the beginning the CenterNet model had higher accuracy than YOLOv3. 

I think that the increase in accuracy is notable in both cases. 
Different choices made during the process paid off even though it was the first time I was using a CenterNet model. 

## Some Results

### Centernet

![CenterNet](/assets/img/posts/beam_counting_p2/CenterNet_steel_1.png)

### YOLOv3

![YOLOv3](/assets/img/posts/beam_counting_p2/YOLOv3_steel_1.png)