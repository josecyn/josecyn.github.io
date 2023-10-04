---
layout: post
permalink: /01ead88355127f33e70117da2578556e80ac1dbc/:categories/:year/:month/:day/:title:output_ext
title:  "How to Use OpenMMLab's mmcls"
description: "State of the Art models without writing a single line of code"
type: card-dated
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
categories: post
tags: 
  - mmcls
  - Pytorch
author: Jose
card: card-1
---


# Purpose

Describe how to use `mmClassification` (also called `mmcls`).
mmcls is a very popular library used in computer vision classification tasks.

- [Repo](https://github.com/open-mmlab/mmpretrain)
- [Documentation](https://mmpretrain.readthedocs.io/en/latest/)

In addition, I would like to mention `mmDetection`.
mmDetection is a very similar framework but used in object detection tasks.

- [Repo](https://github.com/open-mmlab/mmdetection)
- [Documentation](https://mmdetection.readthedocs.io/en/stable/)

Both libraries work in the same way. In this document I will describe `mmcls` but the conceptis the same in
`mmdet`.

At the beginning, the configuration files could seem overwhelming, 
but they offer great flexibility. And they will become easier to read after you modify a few of them.

I just want to explain something specific for `mmdet` regarding the use of XML datasets.
See below the section "Note On mmdet".

Both libraries have amazing documentation! I encourage you to take a look at it!

# How to Install

Installation is easy. Just follow the steps from the official documentation.

[mmClassification Official Documentation - Installation](https://mmpretrain.readthedocs.io/en/latest/get_started.html#installation)

Basically, execute the following commands:

```
# execute in command line 
conda create --name mmlab python=3.8 -y
source activate mmlab
conda install pytorch torchvision -c pytorch -y
pip install -U openmim  
mim install mmcv-full
pip install mmcls mmdet mmengine albumentations
```

# How to Use

This library gives you access to dozens of architectures. 

On top of that, you don't need to write any line to train the model.

Let's see how you can use it.

## Data Format

The most important element. mmcls supports different formats: ImageNet, CIFAR, VOC, etc.

You can transform your annotations to one of those standard formats. 

Another possibility is to use the [CustomDataset class](https://mmclassification.readthedocs.io/en/latest/api/datasets.html#custom-dataset).

I use CustomDataset in the example below.

## Model Architectures

Now it is time to choose your architecture. 

Here is the official documentation: [Model Zoo](https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html)

## Download Configs

mmcls works using configuration files. It is very convenient because you don't have to write 
any piece of code.

I recommend to download a public available configuration file.  

You can download the config file using this command:

```
mim download mmcls --config mobilenet-v2_8xb32_in1k --dest .
```

The command above downloads the configuration file for a MobileNetV2 model ready to be  
trained on ImageNet data on the current directory.

You will find 2 new files in the directory:

- mobilenet-v2_8xb32_in1k.py
- mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth

The .py file is the configuration and the .pth file is the pre-trained weights file.

# Edit the Config File

The config file is ready a specific task but you want to train the model on your data.  

On top of that, you want to apply your own augmentations.

To achieve that you can edit the config file.

Following, you have important sections of the config file.

## Number of Classes

Most models define the number of classes in the architecture definition:

```python
mosel = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0), 
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=12,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0), 
        topk=(1, 5)))
```

Change the num_classes according to your task.

## Input Image Size

In the config file you will have 2 pipelines: 1 for training and 1 for testing.  

Edit these pipelines according to the pipeline you need.

You can change the input image size using the Resize class like this:

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(300, 300)),
    ...
```

## Batch Size

You can edit the batch size in the data section of the config file.

For example, if you want batch size = 64, you write:


```python
data = dict(
    samples_per_gpu=64, 
    ...
```

## Point to Your Dataset

Inside the data dictionary you will have train, validation and test sections.

If you have a CustomDataset, you can point to your data the following way:


```python 
train=dict(
    type='CustomDataset',
    data_prefix='datasets/varieties_split/train',
    ...
```

And in the same way for validation and test.

## Metrics

You can set the evaluation metrics easily creating the following object:

```python
evaluation = dict(save_best='auto', metric='accuracy') 
```

In that case I used accuracy for my image classification model. 

Check the documentation for other metrics.

## Optimizer and LR

Normally you don't have to change this in the configuration file. But if the model suffers from 
exploding or vanishing gradients you can adapt the values of the Optimizer and the LR
scheduler.

Example:

```python
optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict(grad_clip=None)  
lr_config = dict(policy='step', gamma=0.98, step=1)
runner = dict(type='EpochBasedRunner', max_epochs=50)
```

## Save Checkpoints and Loggers

Normally you want to save the best checkpoint during the training. 

Also, logging is important to review the training. You can rely on just text log or also add
Tensorboard.  

Example:

```python
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(  
    interval=25,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
```

# Execute Training

Once you have your config file ready, you can start your training.  

It is as easy as just calling the train.py script on your config file:

```
python train.py configs/mobilenet-v2_simple_split.py
```

The train script can be found in the mmcls repository [here](https://github.com/open-mmlab/mmpretrain/blob/master/mmcls/apis/train.py).

# Note on `mmdet`

Loading your VOC XML dataset when using mmdet can be a bit difficult. 

Debugging is also hard.

This is what worked for me:


```python
data = dict(
    samples_per_gpu=24,  
    workers_per_gpu=4,
    train=dict(
        type='XMLDataset',
        ann_file='../datasets/train/train_files.txt',
        img_subdir='../datasets/train/images/',
        ann_subdir='../datasets/train/annotations/',
        classes=['seal', 'organization'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            ...
    val=dict(
        type='XMLDataset',
        ann_file='../datasets/validation/validation_files.txt',
        img_subdir='../datasets/validation/images/',
        ann_subdir='../datasets/validation/annotations/',
        classes=['seal', 'organization'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            ...
```

In the `train_files.txt` and `validation_files.txt` just write the stem of the files.  

Like this:

```
AP21264_0
AP21470_0 
...
```

Then, you just have to place the images in `images` folder and the annotations in  
`annotations` folder. Please note that at the time of writing, VOC XML Dataset only supports
`.jpg` images.


Using the configuration files, it is very easy to create a YOLOv3 architecture using MobileNetV2 backbone.

# Conclusion

We have seen how to use mmcls and mmdet for image classification and object detection.

These libraries allow you creating state of the art models without writing any line code. Just
editing the configuration files.

Isn't it great!?

