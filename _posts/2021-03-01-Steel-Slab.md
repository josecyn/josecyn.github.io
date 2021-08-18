---
layout: post
title:  "ID Recognition On Steel Slab"
description: "ID Recognition from camera footage"
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

In this project, the customer asked to identify the IDs written on the side of steel slabs.

The images come from the footage of a security camera located in the factory. 

The problem is a bit hard because the steel slabs are piled up together and then moved one by one 
through a kind of a giant conveyor belt. Only the IDs **on a specific place** should be identified. 

## Solution Proposed

In this project, the main problem is to get rid of false detections in the background pile of steel slabs.

For that, I proposed a 2-step algorithm where in the first step the ID would be detected, and in the second 
step the characters would be recognized. 

The task of detecting an ID in its place would be carried by a YOLOv3 tiny model, as it is a quite simple task.
Once the position of the ID is known, a crop around the area will be made.

The cropped image containing the ID will then be passed to a YOLOv3 model, which will identify the characters of the ID.

## Conclusions

That was the first project where I implemented a 2-step algorithm of this kind. It worked very well!

## Results

On top of the image you can see the predicted ID, the expected ID and whether they match (green, OK) or not (red, NG).

![success](/assets/img/posts/steel_slab/slab_2.png)

Of course, this solution is not perfect and due to the low quality video input source it also has some limitations 😉...

![error](/assets/img/posts/steel_slab/slab_3.png)

