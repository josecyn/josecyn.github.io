---
layout: post
title:  "Applied Quantization With Pytorch"
description: "Results of applying Pytorch quantization"
type: card-dated
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
categories: post
tags: 
  - Pytorch
author: Jose
card: card-2
---


## Introduction

In this post, I will review the results of applying quantization to a couple of Pytorch models: YOLOv3 and YOLOv3 tiny.

The YOLOv3 is a very large architecture and the tiny version of it is quite small, so we can compare the effect of quantization in both large and small models.

I converted the original darknet model to ONNX and then I took it as the baseline.
Then, I converted the same model to Pytorch and executed it in FP32 (floating point 32 bits) without any change.
Following, I executed quantization as explained here: [Pytorch Quantization of YOLOv3 Models]({% post_url 2021-06-25-Quantization %}).
Finally, I took the output Pytorch INT8 model and executed it on the chosen devices.

The times that are shown below these lines refer just to the inference time of the model.
It does not include pre or postprocessing. These times are the result of the average of different executions.

## Hardware

I think that quantization is very important for handheld devices in order to improve efficiency.
The devices used to evaluate these models are the Google Pixel 4 and Google Pixel 5.
These devices are very different from each other. The Pixel 4 equips a Qualcomm Snapdragon 855 while the
Pixel 5 has a Qualcomm Snapdragon 765G.

The processor of the Pixel 4 is clearly faster when executing neural networks.
We will see that in the results.

## Results YOLOv3 Tiny

Following, the results of the execution of **YOLOv3 tiny** model on the Pixel 4 and 5.

### Pixel 4

In this use case, we can see that the execution using the Pytorch runtime is 
slower than using the ONNX runtime when both use FP32. 
However, executing the same model in Pytorch using INT8, results in a very short inference time.

When the model just takes 28ms to execute the forward pass, it is possible
to create **real-time applications**.

Concretely, applying INT8 results in almost **2x** improvement regarding ONNX and **2.2x** regarding the FP32 version.

When looking at the F1 accuracy, the baseline is 97%. 
The Pytorch implementations are 91% for FP32 and 95% for INT8. 
However, it is somehow strange that the FP32 implementation has such a big difference.

In that case, though, **the performance of the INT8 is excellent**. It delivers almost
the same F1 accuracy but the execution time is nearly half.

Power consumption of the INT8 version is **1/5** of the ONNX implementation.

![YOLOv3 tiny Pixel 4](/assets/img/posts/quantization_p2/q_yolov3_tiny_p4.png)

### Pixel 5

This is another example where the INT8 version shines. 
The baseline takes 160ms, the Pytorch using FP32 takes 135ms and the INT8 version just takes 37ms.

F1 accuracies are the same as in the previous device, so not a big difference for the  INT8 version.

Execution times are shorter using INT8: **4.3x** and **3.6x** faster than the other implementations.

![YOLOv3 tiny Pixel 5](/assets/img/posts/quantization_p2/q_yolov3_tiny_p5.png)

## Results YOLOv3

Following, the results of the execution of **YOLOv3** model on the Pixel 4 and 5.

### Pixel 4

Starting with the YOLOv3 model on the Pixel 4, we can see that the Pytorch
runtime is faster than the ONNX runtime. 
When executing the Pytorch models, the INT8 version is **1.17x** faster than the FP32 version and **1.40x** faster than the ONNX baseline.

The F1 value of the ONNX model is 97%. 
The F1 value of the Pytorch FP32 is 96% while the INT8 is 93%.

In many cases, that drop in F1 accuracy can be acceptable if inference time is more important.

But we should not just only look at F1 accuracy and inference time. Another important
metric in such devices is power consumption. During my tests, the INT8 version 
consumed almost **half** of what the ONNX baseline does.

![YOLOv3 Pixel 4](/assets/img/posts/quantization_p2/q_yolov3_p4.png)

### Pixel 5

The Pixel 5 is a completely different story. 
The YOLOv3 model is a very big model and if the device does not have
specific hardware for running it, it can be very slow on just CPU.

In that case, the Pixel 5 just has a CPU to run the model, so it is very slow when compared to the Pixel 4.

The ONNX baseline takes 1.7 seconds to execute the forward pass.
The Pytorch versions take 1 second and 0.5 seconds respectively for FP32 and INT8.

Of course, the F1 accuracies are the same as reported in the previous section.
Changing the device does not change the result of the model.
ONNX Baseline 98%, Pytorch FP32 96%, and Pytorch INT8 93%.

In that case, the INT8 version delivers **3.2x** faster inference than the baseline and almost **2x** faster than the FP32 counterpart.

![YOLOv3 Pixel 5](/assets/img/posts/quantization_p2/q_yolov3_p5.png)

## Conclusions

As conclusion, I would like to mention that the performance shown by the
quantization of the YOLOv3 and YOLOv3 tiny models is very satisfactory. 
The impact in F1 accuracy can be OK in some cases and the improvement
in execution time is remarkable.

It is important to note that, as we've seen, it depends on the device.
If your device just supports CPU execution, you can profit massively from 
quantization with improvements **nearly 4x**.

However, if your device has some kind of accelerator, it may not be worth it.
Some accelerators just support FP32 inference. Therefore, when using INT8 you
don't use the accelerator but the CPU. And in that case, the improvement 
can be dimmed, as we saw with the Pixel 4.

In that case, if the accelerator supports it, it could be a good idea to execute
the model in half-precision, FP16.

There is a bigger improvement in YOLOv3 tiny than in the larger YOLOv3. 
The larger YOLOv3 model has more layers that can't be quantized and those 
layers also do a heavier task.
Therefore, the improvement seen in YOLOv3 is not so remarkable as with the YOLOv3 tiny. 
