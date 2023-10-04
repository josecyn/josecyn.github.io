---
layout: post
permalink: /01ead88355127f33e70117da2578556e80ac1dbc/:categories/:year/:month/:day/:title:output_ext
title:  "Object Detection Applied Weak Supervision"
description: "How to practically apply weak supervision to an object detection problem"
type: card-dated
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
categories: post
tags: 
  - Weak Supervision
  - Object Detection
  - Pytorch
author: Jose
card: card-1
---


## Purpose 

This document explains how we built a computer vision model to detect seals and organizations in invoice documents. The goal is to provide a useful explanation for engineers interested in building similar computer vision models by detailing the thought process and design choices.

## Introduction

We want to detect organizations in PDF invoice documents, both searchable and unsearchable. The solution should perform well on both document types. 

Organization information is usually located in the top right of an invoice, like this:

![Sample invoice](/assets/img/posts/applied_weak_supervision/invoice.png)

The green box shows the bounding box we want the model to detect. 

Invoices usually have a standard format. Unsearchable PDFs are challenging for NLP as all text is embedded in the image. Therefore, we considered transforming PDFs into images and applying computer vision, which handles multiple organizations and is fast.

Japanese invoices sometimes have seals over the organization name. In these cases, OCR doesn't work because the name is obscured. However, we developed an image process to remove the red channel and reveal the name, increasing OCR accuracy.

Because of this seal problem, we also decided to detect seals to apply the image process and remove them.

## First Step: Proof of Concept

I like to start small with a proof of concept and iterate. From experience, I was confident seal detection would work but unsure about organizations since placement varies. This variance challenges computer vision models. 

We selected some invoices, labeled ~350 images, and quickly trained a model to evaluate feasibility. We used ~280 images for training and reserved some for validation and testing to check for overfitting.

I chose YOLOv5 for its speed in training and inference plus my previous experience. YOLO models tend to be fast and accurate.

### PoC Results

Here are the precision, recall and F1 scores on validation and test sets:

**Validation**

| Class        | precision | recall | F1   |
| ------------ | --------- | ------ | ---- |
| Seal         | 0.97      | 0.97   | 0.97 |
| Organization | 0.66      | 0.93   | 0.77 |

<br>

**Test**

| Class        | precision | recall | F1   |
| ------------ | --------- | ------ | ---- |
| Seal         | 0.97      | 0.94   | 0.95 |
| Organization | 0.48      | 0.64   | 0.55 |

<br>
As expected, seal detection performed very well. Organization results were mixed but over 50% accuracy on challenging data with limited training is good for a PoC.

Unsuccessful results would have been under 25%, indicating the problem is too difficult for this approach. The next step would have been to try a larger model. If that still failed, the problem likely requires a different solution.

The output images showed introducing negative examples could make the model more robust to false positives. 

This successful PoC gave us confidence to continue exploring this solution.

## Weak Supervision

With thousands of PDFs available, organization detection can likely be improved. However, labeling at scale is expensive. 

We already have a 50% accurate model. We can use it to help label additional data faster by correcting its annotations.

More on the description of weak supervision workflow on another blog post.

### First Iteration

As this was the first iteration, we didn't label all available images. The PoC used ~350 images, so we tripled that to ~1000, covering more invoice formats and likely improving the model. Correcting the model's labels takes seconds per image. We also added unlabeled negative examples.

We ended up with ~950 training and ~75 validation images. No test set yet. 

During the labeling, we expanded the organization bounding boxes, which were too small, omitting info like phone numbers. My experience suggests that's related to network size, and more parameters can better infer box sizes.

**First iteration results**

| Class | precision | recall | F1   |
|-----|---------|------|----|
| Seal  | 0.97      | 0.96   | 0.96 |
| Organization  | 0.84      | 0.90   | 0.87 |

<br>
Fantastic! Even with a smaller model, organization accuracy jumped to ~90%. This validates that more data improves the model.

### Second Iteration 

Now we validate the process with a larger model and test set. We build the final model but with less data.

I converted all 10,000+ PDFs to images but only used 2000, shuffled without previous annotations. Since the model is very accurate already, correcting these labels took ~2 hours.

- Training: 1600 images  
- Validation: 200 images
- Test: 200 images

Based on the experiments, this should further improve results. 

I added image augmentation to reduce overfitting on static training boxes.

**Second iteration results**

| Class        | precision | recall | F1   |
| ------------ | --------- | ------ | ---- |
| Seal         | 0.99      | 0.97   | 0.98 |
| Organization | 0.92      | 0.88   | 0.90 |

<br>
Great results! 90% accuracy on a proper test set likely reflects real-world performance. Bounding box sizes also look good. 

The model seems ready. Final steps are labeling all images, balancing searchable/unsearchable data, and retraining. Active learning could help select the best data to annotate.

## Conclusion 

We went from ~50% accuracy to ~90% on the hardest class by using 20% of the available data through an iterative process.

Gains in machine learning come more from this process and data quality than state-of-the-art architectures. 

The full model was built using an AWS ml.g4dn.xlarge SageMaker instance, training for ~90 minutes with early stopping. At ~$1/hour, the total compute cost was ~$1.50. Deep learning doesn't always require massive data and resources.

Out of curiosity, I also tried an EfficientDet D3 model:

| Class        | precision | recall | F1   |
| ------------ | --------- | ------ | ---- |
| Seal         | 0.99      | 0.97   | 0.98 |
| Organization | 0.93      | 0.93   | 0.93 |

<br>
EfficientDet improved organization recall to 93% but inference is slower:

| Architecture | Backbone | WeightsSize (MB) | WeightsLoad Time(s) | InferenceTime 1x (s) | Inference Timeavg 200x (s) |
| ------------ | -------- | ---------------- | ------------------- | -------------------- | -------------------------- |
| YOLOv5       | medium   | 141              | 0.49                | 0.72                 | 0.65                       |
| EfficientDet | D3       | 50               | 0.4                 | 1.2                  | 1                          |

<br>
This shows the value of trying different architectures with little code change thanks to icevision.
