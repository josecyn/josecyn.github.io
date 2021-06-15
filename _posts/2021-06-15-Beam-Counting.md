---
layout: post
title:  "Steel Beam counting"
description: "Counting of steel beam objects"
type: card-dated
date: 2021-06-15 07:17:00 +0900
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
last-updated: 2021-06-15 07:17:00 +0900
categories: post
tag: Pytorch
author: Jose
card: card-2
---

### Introduction

The task in this project was to count the number of beams in the image.

At the beginning of the project, I was aked to do segmentation and then count the pieces based on the 
shapes segmented.

After that, I proposed to do object detection as I thought it could lead to higher accuracy.

In order to perform object detection I chose 2 different architectures: CenterNet and YOLOv3.

Let's go through each step of the project.

### Segmentation

I used the famous [Mask RCNN](https://github.com/matterport/Mask_RCNN) repository. At this point, the 
dataset was even less than 1000 pictures. For 