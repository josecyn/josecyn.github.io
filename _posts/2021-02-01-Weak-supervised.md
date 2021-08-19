---
layout: post
permalink: /01ead88355127f33e70117da2578556e80ac1dbc/:categories/:year/:month/:day/:title:output_ext
title:  "Weak Supervision Dataset Creation"
description: "Fixing a YOLOv3 model using weak supervision"
type: card-dated
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
categories: post
tags: 
  - Computer Vision
  - Object Detection
  - Weak Supervision
author: Jose
card: card-1
---

## Introduction

In this project, the customer asked to fix an existing YOLOv3 model that didn't work:

![Need fix](/assets/img/posts/number_recognition/fix_char_recog.png)

Due to the poor performance (as you can see in the image), we decided to begin from scratch 
instead of taking the existing labels and model.

Let's begin by explaining the labeling part.

## Data Labeling

I teamed up with a colleague on this project. We had to label thousands of images so I proposed to use 
a common technique: **weak supervision**. Instead of labeling all the images first and then train a model, 
we could do everything in parallel and save time.

So we first labeled some images, then trained a small model. That model wasn't perfect because we just labeled some data. But it could help us label all remaining data in just a few minutes.

Of course, the new labeled data by the model was not perfect. There were many mistakes, but there were also many hits.

So we iterated a few times correcting some of the mistakes and training the model again until we had the dataset perfectly labeled.

This technique saved us a lot of time. We could label and train a high accuracy model in just a few days!

## Model Development

One of the requirements of this project was to prepare a YOLOv3 model. We already had some experience with this 
architecture so we also iterated finding the proper parameters.

## Conclusions

One of the outcomes of this project was to be able to successfully apply **weak supervision**.
Additionally, even though I am aware of how well YOLOv3 learns from data, I was surprised by the high accuracy that it reached. The large amount of data helped a lot.

## Results

Some examples:

![Result](/assets/img/posts/number_recognition/fix_char_recog_3.png)
