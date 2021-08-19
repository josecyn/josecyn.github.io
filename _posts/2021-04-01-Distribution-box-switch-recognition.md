---
layout: post
permalink: /01ead88355127f33e70117da2578556e80ac1dbc/:categories/:year/:month/:day/:title:output_ext
title:  "Distribution Box Switch Recognition"
description: "Status recognition of distribution box switches"
type: card-dated
image: /assets/img/posts/steel_slab/banner.jpeg
caption: 
categories: post
tags: 
  - Computer Vision
  - Object Detection
author: Jose
card: card-3
---

## Introduction

This project is about creating a demo of object detection. 
In this case, the objects are switches in a distribution box. 
And it is not only about finding switches in the image but also detecting their status: **on or off**.

2 different models of distribution boxes were supported. One was more industrial and the other 
was a model very common in homes in Japan.

![Toshiba AR100 smartglasses](/assets/img/posts/distribution_box/dbox_1.jpg)


## Hardware

The hardware used in this project is Toshiba's smartglasses.
These smartglasses have a small screen with a resolution of 640×360 and a 5MP camera with autofocus.
They are connected via USB-C to a portable mini PC with Windows 10 OS and Intel Core m7 mobile processor.

For more info: [Toshiba AR100](https://dynabook.com/business/mecd/product/ar100-mar-2019/index.html).

![Toshiba AR100 smartglasses](/assets/img/posts/distribution_box/smartglasses.jpg)

## Dataset Preparation

In this project, the dataset was created by myself. The time spent labeling a single image of the bigger 
distribution box was long, so it was important to use as few images as possible.

It was also important to take those images using the smartglasses instead of another camera, 
so the training images are as similar as possible to the actual images that will be used during the demo.

In pictures, illumination is key. So taking the images in the same area where the demo will take place, was also crucial.

In total, **less than 200 images were used**.

## Model Development

This project was the first I used YOLOv3 Tiny. The mini PC processor is quite slow, therefore the model's small size and fast inference time were ideal for this task. 

One of the hardest things about this project was that the switches are so small and so close to each other, 
that the input resolution to the model had to be very big: **608x608 pixels**. 
Otherwise, the model accuracy would drop dramatically.
Additionally, the proper tunning of NMS parameters was very important.

## Application Development

In this demo, the C++ application running on the smartglasses played a very important role. 

In the first part of the demo, I showed how the model recognizes the different switches on both distribution boards 
also been able to detect its status: on / off.

In the second part of the demo, the user will scan a QR code. 
This QR code represents the expected configuration of the distribution board.
Next, after scanning the QR code, the user looks at the distribution board and 
the switches **mismatching** the expected configuration are shown in blue as a mistake in the current configuration.
When the user changes the status of the switches and the current configuration matches the expected configuration
a green "○" is displayed.

This demo shows a possible application in the industrial field, where the operator
goes to a customer to check its distribution board and he can use the smartglasses to **double-check** the 
configuration. As there are many small switches, it is an error-prone task.

It can also be applied at home, where the user with its smartphone can check the proper configuration without 
any other intervention.

The first version of the application was not very smooth, running at 2~3FPS.
In the next section, I explain the process of optimizing the model.

## Model Optimization

During the investigation on how to improve the model, I found a repository called [SlimYOLOv3](https://github.com/PengyiZhang/SlimYOLOv3).

This repository is based on the paper [SlimYOLOv3: Narrower, Faster and Better for Real-Time UAV Applications](https://arxiv.org/abs/1907.11093).

In this paper, the authors explain a way of pruning YOLOv3 models to make them smaller and therefore faster to execute. Reducing inference time to make the model able to run real-time applications.

The pruning procedure runs in three main steps:

1. Sparsity training: learn what parts of the network are more important.
2. Prune the network: remove less important parts of the network.
3. Fine-tuning: retrain to let the network recover from the pruning.

These 3 steps can be executed more than once.

In fact, in this case, I could run this process 3 times achieving at the end a speed-up of **more than 2x**.

The final version of the application runs smoother than the first at **6FPS** with an input resolution of **672x672 pixels**.
I had to increase the input resolution to keep high accuracy. 
The drop in accuracy was just a bit when compared with the original model.

## Conclusions

I enjoyed this project a lot. I was responsible for everything from data collection, labeling, 
model development and optimization, application design and development, and demonstration.

Model optimization was a hard task but it paid off. We were very pleased with the model's improvement.

## Results

In this video you can see the final version directly recorded from the smartglasses.
It shows the switch recognition running at 6FPS with an input resolution of 672x672 pixels.
On switches are shown in green, of switches are shown in red.

<!-- blank line -->
<figure class="video_container">
  <video controls="true" allowfullscreen="true">
    <source src="/assets/img/posts/distribution_box/video.avi" type="video/avi">
  </video>
</figure>
<!-- blank line -->

