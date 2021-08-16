---
layout: post
title:  "Pytorch Quantization Of YOLOv3 Models"
description: "The theory behind applying quantization to YOLOv3 and YOLOv3 tiny"
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

In this document, I will go through the process of quantizing an existing YOLOv3 model using Pytorch.
Please note that nowadays, in May 2021, Pytorch does not support quantization using GPU!

Quantization refers to convert input, weights, and biases to floating-point 16 or integer 8 bits.
Operations in floating-point 32 bits have additional precision that isn't needed for the model to have high accuracy.
Because of that, the use of smaller precision increases execution speed while keeping a good accuracy.

However, when converting to integer 8, the impact is obviously bigger and some steps have to be taken in order to preserve the model's accuracy.

In this guide, we will take a look at those steps. We will focus on int8 because that is
where you can decrease the execution time the most.

## First Steps

The idea of quantization is to convert the input of the model to int8, execute the forward pass
and convert the result back to floating-point.
This approach is fine, but sometimes the model may contain operations that do not support int8 input.

Check the following link to have a look at the [supported operators](https://pytorch.org/docs/stable/quantization-support.html).

If you use a simple model such as MobileNet or even ResNet, all operators are supported and you don't have to change the code that much.

As stated in the introduction, nowadays, Pytorch only supports quantization using CPU. Because of that, you have to
load or move the model to the GPU.

In the case of YOLOv3 and YOLOv3-tiny, I found that I was using `leakyRELU` which is not supported.
So I replaced the `leakyRELU` with plain `RELU`.
After this replacement, I retrained the model for a few epochs just to adapt to the new component.
This is an important step! Afterward, check the model's accuracy, it should not have changed that much.

Of course, there may be other operators that are not supported and can't be changed so easily.
In the case of YOLOv3, you can see that the YOLO layer is, of course, not supported.
Moreover, in the case of the YOLOv3-tiny version I used, the operation `ZeroPad`, which is not supported.

Let's see in the next chapter how can we deal with such cases.

## Unsupported operations, what to do

Other than replacing the operator with a similar one as we did with `leakyRELU` and `RELU`,
there is another possibility, but first, we need to know how quantization internally works.

When we quantize weights, biases, and operators, we switch from fp32 data to int8 data.
When executing the `forward` pass, if one operator expects fp32 data and receives int8 data, it will fail and vice versa will also fail.
That's why Pytorch offers the possibility to convert data from/to int8 and fp32.
The name of functions for that purpose is: `DeQuantStub` and `QuantStub`.

So, if we have an unsupported operator, the only thing you need to do is convert its input data from int8 to fp32.
Of course, this conversion takes time and memory so, in terms of performance, the least changes the better.

To have a clear view, it would be something like this:

![Iterative forward](/assets/img/posts/quantization/dequantstub_quantstub.png)

Note that between supported operations you don't need to change the data type.

Let's see an example of how to control this flow programmatically.

```python
def __init__(self):
    self.quant = torch.quantization.QuantStub()
    self.dequant = torch.quantization.DeQuantStub()
    self.module_list = # create model
    ...


def forward(self, x):
    yolo_outputs = []
    noquant_layers = self.get_noquant_layers()        # get no quant operations index

    x_isquant = False
    for i, module in enumerate(self.module_list):
        # quant control
        if i in noquant_layers and x_isquant:
            x = self.dequant(x)
            x_isquant = False
        elif i not in noquant_layers and not x_isquant:
            x = self.quant(x)
            x_isquant = True

        x = module(x)

    return x
```

In this example, I get the "no quant layers" at the beginning calling the method `self.get_noquant_layers()`.
It returns a list with the unsupported layer's index.
Those layers need fp32 input, so if the data is in int8 format, we need to convert it.
That's why I added a "quant control" block.
The quant control block checks whether the module that's about to be executed needs fp32 or not.
In case the module needs fp32 input, and the input is int8 type, we convert it.
In case the module needs int8 and the data is of type fp32, we also convert it back to int8.

For those conversions, You have to create 2 properties in the model constructor `__init__(self)`.
Note the `self.quant()` and `self.dequant()` methods.

## Fusing Operations

Now, that the forward flow is correct, let's take it a step further.
Pytorch has a few already optimized operators that include multiple operations inside just one operation.
For example `ConvReLU2d` includes `convolution + relu` operations.
You can find these operations in this [list](https://pytorch.org/docs/stable/quantization-support.html#torch-nn-intrinsic).

So, using these combined operations we can extract a little bit more of performance.
Pytorch provides a function for that task that is called `torch.quantization.fuse_modules()`.
This is an example on how to use it:

```python
def fuse_model(self):
    for i, m in enumerate(self.module_list):
        if isinstance(m, nn.Sequential) and isinstance(m[0], nn.Conv2d) and len(m) >= 2:
            torch.quantization.fuse_modules(m, [['conv', 'batch_norm', 'relu']], inplace=True)
```

In the code above, we go through all the sequential modules checking whether the first element is a convolution layer.
If that is the case, we call the Pytorch method with the keys we use in our model `['conv', 'batch_norm', 'relu']`.
Pytorch will take those operations and merge them into a single one such as `ConvReLU2d` (Conv2d + ReLU).

Example before fusing:

```
(0): Sequential(
      (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (batch_norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
    )
```

After fusing:

```
(0): Sequential(
      (conv): ConvReLU2d(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
      )
      (batch_norm): Identity()
      (relu): Identity()
    )
```

Note the use of the `Identity()` operation in the replaced operators `batch_norm` and `relu`.

After fusing those operators, execute a validation on the model to check that the accuracy didn't decrease.
The model accuracy should be the same.

## What To Do With Skip Connections

In `ResNet` networks and many other architectures, we may use skip connections. In YOLOv3 architecture we do, and it needs to be handled.
If you just concatenate 2 tensors using `torch.cat()` and both tensors have the same datatype (int8 or fp32) it should be fine.
However, when skipping connections the addition `+` operator is often used.
In that case, we need to update this operator to work with `int8` input.

To achieve that, you have to declare a new property in your class like this:

`self.f_add = nn.quantized.FloatFunctional()`

It has to be at the class level because this operation will be calibrated so it has to save statistics. It can't be created 'on the fly'.

When executing the addition operation, you can use it like this:

`x = self.f_add.add(x, previous_x)`

Of course, you have to pay attention that the data types of the tensors `x` and `previous_x` match!

This is how it looks when the model structure is printed:

```
(17): Connect(
  (f_add): FloatFunctional(
    (activation_post_process): Identity()
  )
)
```

## Different Quantization Methods

Once the model is ready, we can start the process of post-training quantization.

Let's see 2 options:

The **post-training static quantization** method involves not just converting the weights from `float` to `int`, as in dynamic quantization,
but also performing the additional step of first feeding batches of data through the network and
computing the resulting distributions of the different activations.

**Post-training quantization-aware training (QAT)** is the quantization method that typically results in the **highest accuracy**.
With QAT, all weights and activations are “fake quantized” during both the forward and backward passes of training:
that is, `float` values are rounded to mimic `int8` values, but all computations are still done with floating-point numbers.
Thus, all the weight adjustments during training are made while “aware” of the fact that the model will ultimately be quantized;
after quantizing, therefore, this method will usually yield higher accuracy than either dynamic quantization or post-training static quantization.

## How To Set The Quantization Configuration

There are currently 2 different configurations for the quantization process:
`fbgemm` (for use on x86, [https://github.com/pytorch/FBGEMM](https://github.com/pytorch/FBGEMM)) and `qnnpack` (for use on the ARM QNNPACK library [https://github.com/pytorch/QNNPACK](https://github.com/pytorch/QNNPACK)).

In this case, I am targeting ARM processors, so the best configuration to choose should be `qnnpack`.
However, I would recommend you to try both methods.

The code to set the configuration and prepare the model couldn't be easier, but it depends on the quantization method:

```
qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)
```

However, if you are using **quantization aware training** method, you have to call this function:

```python
model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
torch.quantization.prepare_qat(model, inplace=True)
```

When printing the configuration, the output looks like this (QAT training):

```
QConfig(activation=functools.partial(<class 'torch.quantization.fake_quantize.FakeQuantize'>, observer=<class 'torch.quantization.observer.MovingAverageMinMaxObserver'>, quant_min=0, quant_max=255, reduce_range=False), weight=functools.partial(<class 'torch.quantization.fake_quantize.FakeQuantize'>, observer=<class 'torch.quantization.observer.MovingAverageMinMaxObserver'>, quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False))
```

If you want to check manually if the observers were correctly placed, you can just print the module like this:

```python
print(model.module_list[0][0])
```

It prints the information of the first element in the first layer (static quantization).

Output:
```
ConvReLU2d(
 (0): Conv2d(
   3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
   (activation_post_process): HistogramObserver()
 )
 (1): ReLU(
   (activation_post_process): HistogramObserver()
 )
)
```

Note the `HistogramObserver()` functions.

And this is how it looks for QAT:

```
ConvReLU2d(
  3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
  (activation_post_process): FakeQuantize(
    fake_quant_enabled=tensor([1], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), scale=tensor([1.]), zero_point=tensor([0])
    (activation_post_process): MovingAverageMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
  (weight_fake_quant): FakeQuantize(
    fake_quant_enabled=tensor([1], dtype=torch.uint8), observer_enabled=tensor([1], dtype=torch.uint8), scale=tensor([1.]), zero_point=tensor([0])
    (activation_post_process): MovingAverageMinMaxObserver(min_val=tensor([]), max_val=tensor([]))
  )
)
```

Note the `FakeQuantize()` functions.

## Post-training Quantization General Steps

basic steps to perform post-training static quantization:

1. Load model
2. Evaluate the model to get the baseline accuracy
3. Fuse possible operations (for example conv, bn and relu)
4. [Optional] check the baseline accuracy again (should be the same as in step 2)
5. Set quantization configuration
6. Calibrate / QAT train
7. Convert the model to a quantized model
8. Evaluate the model to check accuracy regarding baseline
9. Save the quantized model

In step 6, depending on your quantization method, you will calibrate or do QAT training.
Read the following sections to understand the difference between them.

### Post-Training Static Quantization Calibration

In the process of calibration, we set the model in `eval()` mode and execute inference using training data.
Pytorch captures the distributions of the activations automatically.

In the calibration process, you can use a subset or the whole training set. And you can even do multiple epochs.
I would test different options and compare the results.

So, there's nothing else to add about the calibration step, you just have to do inference. Just keep in
mind to use a representative set of data of the distribution of your dataset.

Let's see how the quantization-aware training differs from calibration in the next section.

### Post-training Quantization Aware Training

In static quantization, we did a calibration step where we executed inference.
In QAT training, you just have to train the model as you would normally for a few epochs.
During forward and backward Pytorch will collect statistics that will use later during quantization.
Additionally, you can also do some tricks to try to increase accuracy:

* Switch batch norm to use running mean and variance towards the end of training to better match inference numerics.
* We also freeze the quantizer parameters (scale and zero-point) and fine-tune the weights.

Example of training loop:

```python
for epoch in range(args.cal_epochs):
    print(f'QAT Training process epoch {epoch+1}/{args.cal_epochs}')
    model = train(model, opt, conf, data_loaders['train'])    # training epoch
    if epoch > 3:
        # Freeze quantizer parameters
        model.apply(torch.quantization.disable_observer)
    if epoch > 2:
        # Freeze batch norm mean and variance estimates
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.quantization.convert(model.eval(), inplace=False)
    metrics = test(quantized_model.eval(), args, loaders['test'])
    print(f"## QAT Model Evaluation Results Epoch {epoch+1}##")
    pprint.pprint(metrics)

print('QAT Training done')
```

Again, I encourage you to test different parameters.
Please note that this process can be veeeeery slow for big datasets or many epochs.

## Save model

When saving for using in a mobile device, for example, you have to do it in `jit` format.
For example:

```python
def get_traced_module(model, img_size) -> torch.nn.Module:
    input_data = get_random_input(img_size, model.gpu)      # get_random_input just uses torch.randn
    return torch.jit.trace(model.forward, input_data).eval()

torch.jit.save(get_traced_module(quantized_model, imsize), args.export_file)
```

## Summary Whole Process Step-by-Step

1. Train a floating-point model or load a pre-trained floating-point model.
2. Move the model to CPU and switch the model to evaluation mode.
3. Apply layer fusion and check if the layer fusion results in a correct model.
4. Apply `torch.quantization.QuantStub()` and `torch.quantization.QuantStub()` to the inputs and outputs, respectively.
5. Specify quantization configurations, such as symmetric quantization or asymmetric quantization, etc.
6. Prepare a quantization model for post-training calibration.
7. Run post-training calibration.
8. Convert the calibrated floating-point model to a quantized integer model.
9. [Optional] Verify accuracies and inference performance gain.
10. Save the quantized integer model.

## Mobile Optimization

Apart from using `qnnpack` configuration, you can also take a few other measures.
I particularly like this link: [https://pytorch.org/tutorials/recipes/mobile_perf.html](https://pytorch.org/tutorials/recipes/mobile_perf.html)

Check it out!

## Conclusion

Using quantization techniques you can reduce the model's execution time while
having a close accuracy to the baseline. Execution speed can be 2x-4x faster (depending on
multiple factors) and the model size will be reduced too because the weights are stored directly in int8
taking less space in your memory.

## Links

* [https://pytorch.org/tutorials/recipes/mobile_perf.html](https://pytorch.org/tutorials/recipes/mobile_perf.html)
* [https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
* [https://pytorch.org/docs/stable/quantization.html](https://pytorch.org/docs/stable/quantization.html)
* [https://leimao.github.io/blog/PyTorch-Static-Quantization/](https://leimao.github.io/blog/PyTorch-Static-Quantization/)
* [https://pytorch.org/docs/stable/quantization-support.html](https://pytorch.org/docs/stable/quantization-support.html)



