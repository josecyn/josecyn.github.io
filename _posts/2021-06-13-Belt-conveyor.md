---
layout: post
title:  "Handwritten character recognition"
description: "Hand written character recognition in any direction on a conveyor belt."
type: card-dated
date: 2021-06-13 15:49:00 +0900
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
last-updated: 2021-06-13 15:49:00 +0900
categories: post
tag: computer vision
author: Jose
card: card-2
---


### Introduction

In this project I created a demo of handwritten character recognition.

In the team we had a previous experience working on a problem like this, and I decided to showcase
that solution.

The idea of this demo is to use a robot arm and a belt conveyor to simulate a industrial environment where 
this technology could be used. The robot arm picks up a box with handwritten characters and places it on the belt conveyor.
A camera located next to the belt, is recording video all the time. Using deep learning models the text on the side
of the boxes is detected, recognized and the result is added to the list on the right.

Handwritten character recognition is not a complex problem, so I decided to make it more
difficult including that the text can be in any direction and that it must run at a decent speed on a 
NVIDIA Jetson.

This is the result:

![Conveyor belt](/assets/img/posts/belt.gif)

It runs at real time 30fps and when it detects a pice of text it runs at 7fps. Still smooth enough.

The data has been manually collected and labeled. 
There were few examples, so this also adds up in the global difficulty of the project.

### Methodology

This is a 2-step process:

* Detect text in the image, crop and rotate
* Apply character recognition to the result of the first step

### Motivation

In this demo I wanted to support any alphanumeric character (plus hyphen) in any direction (up, right, down and left).
At first I wanted to solve this problem with a single model, but soon I realised that it is too complex.

If you have one class for each character in any direction you end up with many classes, around 150.

So I decided to split the detection of the text and the character in two different models.
The first one will carry text detection. When it detects some text in the image, it will crop around it 
and it will rotate to the up position.

Once the text has been cropped and rotated, the second model will carry over and perform character recognition.

### First Model: Text Detection

The first task is to detect text. This is a classic object detection problem with 4 classes: up, right, down and left.

At first it seemed quite easy, so I decided to apply one of the fastest architectures in terms of inference time: **YOLOv3 tiny**.

I have a lot of experience with YOLOv3 tiny and the results were very good from the very beginning. 
I tried to keep the resolution as low as possible and endeup with a input image size of 288x160 pixels.

A couple of examples:

![Conveyor belt text detection ex1](/assets/img/posts/belt_conveyor/belt_text1.png)

![Conveyor belt text detection ex2](/assets/img/posts/belt_conveyor/belt_text2.png)

It is a very fast model, it runs at 30fps on this hardware 
so I was happy with the result and moved to the next subproblem; **character recognition**.

### Second Model: Character Recognition

This is a much harder problem. 
I tried to apply the same architecture as in the previous step, but it didn't work so well.

So I decided to apply a bigger network: **YOLOv3**. As it is a bigger network, it can understand more variations of 
the characters so it can generalize much better.

After working on the parameter tunning, I got a very good model. 

One of the key parameters was the input **image size**. Inference time is very important in this project because we have 
very limited resources. We had 16:9 images, so it was key not to add padding to the images 
(this could increase inference time by 30%). It was also very important to keep a low resolution. 
I got a working model with high accuracy and then iterated until I reached a good accuracy at the smallest resolution.
In that case the final input image size was 384x192 pixels.

A couple of examples:

![Conveyor belt character recognition ex1](/assets/img/posts/belt_conveyor/belt_char1.png)

![Conveyor belt character recognition ex2](/assets/img/posts/belt_conveyor/belt_char2.png)

As expected, this model is slower than the text detection one.
In order to improve inference time, I skip some frames, I don't perform character 
recognition of every frame detected. This could have been achieved increasing the batch size with similar results.

### Additional Information

Additionally to the deep learning models I also developed the desktop application.
The application was made in `Python` using `tkinter` library.

The Robot arm is a [Magician Dobot](https://www.dobot.cc/dobot-magician/product-overview.html) and it is controlled using the Python API provided by the vendor.

The camera is a simple logitech webcam. 

Everything runs on a NVIDIA Jetson TX2.

The models were trained using [AlexeyAB's darknet repository](https://github.com/AlexeyAB/darknet).

I also tried to run those models on `TensorRT` and I succeded. But because of time constraints I decided to 
implement the whole program in `Python`.

If you have questions regarding this project, don't hesitate to contact me!
