---
layout: post
permalink: /01ead88355127f33e70117da2578556e80ac1dbc/:categories/:year/:month/:day/:title:output_ext
title:  "Serving Multiple Models with One SageMaker Endpoint"
description: "Deploying multiple machine learning models to a single endpoint simplifies management and reduces costs"
type: card-dated
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
categories: post
tags: 
  - AWS
  - Sagemaker
  - Pytorch
author: Jose
card: card-1
---


# Introduction

Multi-model deployment is an increasingly popular way to manage and deploy machinelearning models. Rather than deploying a separate endpoint for each individual model, multi-model deployment allows you to deploy multiple models to a single endpoint, making it easier to manage and scale your models.
Amazon SageMaker, a leading machine learning platform from AWS, supports multi-modeldeployment.

In this article, we'll explore how to implement and use this capability in SageMaker, including best practices for managing multiple models.

# Architecture

As you can see in the image below, instead of having one dedicated endpoint for each model, you can have multiple models in just one instance.

![Multimodel diagram](/assets/img/posts/multimodel_sagemaker/multi-model-endpoints-diagram.png)

We are going to use AWS Multi-model server. Multi Model Server (MMS) is a flexible and easyto use tool for serving deep learning models trained using any ML/DL framework.

This deployment style has many benefits. See below the "Advantages" section for more details.

There are other alternatives like NVIDIA Triton server.

# How to Implement Multi-Model Deployment in Amazon SageMaker

Implementing a multi-model deployment in Sagemaker is an easy process that can be done ina few steps.
Let's see them one by one.

## Creating The Docker Image

First of all, we need to create a Docker image that will be deployed.

I started from the official Ubuntu image ubuntu:20.04. Then, it is very important to tell Sagemaker that your container supports multi-model deployment. 
You can do that setting the 2 following labels:

```
# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
```

We are going to build an AWS Multi-model server. So next, we have to install all the required libraries and packages. 3 important required libraries for the server are:
- multi-model-server
- sagemaker-inference
- retrying

Additionally, I need other packages related to the nature of the models I will deploy. In my case, I install PyTorch, Donut, IceVision, and their respective dependencies.

## Creating The Models

Creating the models requires a few steps that need to be done for each model.
First, you need to write the source code to load the model, preprocess the input data,inferencing, and postprocessing predicted results.
These functions need to be implemented in a `Handler Service`. This handler serviceextends the default one called
`sagemaker_inference.default_handler_service.DefaultHandlerService`.
Here's a guide to implement this service and an example implementation.
Next, you need to archive all the source files and the model files using torch-model-archiver.

More info about this tool in the [docs](https://github.com/awslabs/multi-model-server/blob/master/model-archiver/README.md).
The command is as follows:

```
$> torch-model-archiver --model-name [model_name] --version [model_version] --handler [handler_service_file] --extra-files [model_dir] --archive-format no-archive --export-path [output_path]
```

The `archive-format`` is set to no-archive so this will just create a folder with all the files.
It will create a MAR-INF folder with a `MANIFEST.json` file with a content similar to this:

```
{
"createdOn": "16/02/2023 07:25:48",
"runtime":"python",
"model":
{
  "modelName": "donut",
  "handler": "handler_service.py",
  "modelVersion": "1.0"
},
"archiverVersion": "0.7.1"
}
```

The output folder created by the archiver tool will look like this:

```
+-- code 
  - model.py 
  - my_utils.py 
  - requirements.txt
+-- MAR-INF 
  - MANIFEST.json
  - handler_service.py
  - model_file1
  - model_file2
  - ...
```

Now, you can pack the folder created by `torch-model-archiver` into `[model_name].tar.gz`

You have to repeat this process for each model you have!

Finally, you have to upload all the `.tar.gz.` files to the same path in `S3`.

This path will be important later, write it down!

## Creating The Sagemaker Model

This is an easy step where you create the model in Sagemaker.
I used the python SDK for that. See details below.
You need to change the container URI, S3 path and you can change the model name.

## Creating The Sagemaker Endpoint Configuration

In this step we create the configuration of the endpoint. We select the instance type and its count.
It is important that the `model name` matches with the previous step!

## Creating A Multi-Model Endpoint

Finally, creating the endpoint. We connect the model and the configuration of the previous steps.
Pay attention to the names!

## Invocation

Below you can find an example invocation. In the invocation method you have to specify the
`TargetModel`. `TargetModel` is the filename of the model you want to use with this input.
This parameter tells the multi-model server which model to use.

# Results

Following the steps above, I could deploy 3 models into one single endpoint using a
`ml.g4dn.xlarge` instance.

I deployed the following models:
- `YOLOv5m` model with input image size of 640x640 pixels.
- `Donut` model with image input size of 960x640 pixels.
- `Donut` model with image input size of 1280x960 pixels.

The multi-model server adds a bit of overhead on top of the standard execution time of eachmodel. But it is a really small overhead considering data transfer between client and endpoint.
I ran some "stress" tests and the server holds pretty well.
I created a python script that uses multithreading to call the server non-stop on all 3 models.It would produce around 20% to 30% slowdown.
During this stress test, a few exceptions were thrown by the server during the process ofthousands of requests.
The server reached around 92% memory utilization at the beginning but then it was reducedto 80%.
Overall, I think the model can hold pretty well.

# Advantages

1. **Simplified Management**: Multi-model deployment allows you to manage multiplemodels with a single endpoint, making it easier to maintain and scale your machinelearning models. This reduces the need for managing and maintaining separate endpointsfor each model.
2. **Cost Savings**: Deploying multiple models to a single endpoint reduces the number ofinstances required to serve multiple models. This results in cost savings as you do notneed to pay for multiple instances.
3. **Increased Flexibility**: With multi-model deployment, you can serve different models todifferent applications, and you can also use multiple versions of the same model at thesame time. This makes it easier to experiment with different models, compare theirperformance, and implement improvements.
4. **Simplified Deployment**: Multi-model deployment allows you to deploy multiple modelsat the same time with a single click, making it easier to deploy and test new models inproduction. This reduces the time and effort required to deploy new models.

Overall, multi-model deployment in Amazon SageMaker offers several benefits that make iteasier to manage and scale your machine learning models. It reduces costs, improvesperformance, and provides increased flexibility and agility, which are key advantages intoday's fast-paced, dynamic business environment.


# Conclusion

Multi-model deployment is a powerful capability in Amazon SageMaker that simplifies themanagement and deployment of machine learning models. By deploying multiple models to asingle endpoint, you can reduce costs, improve performance, and increase flexibility.
With SageMaker's built-in support for multi-model deployment, it's easy to set up andmanage multiple models in production. 

By following best practices, you can make the most ofthis capability and streamline your model deployment process.

I hope this article has given you a good introduction to multi-model deployment in SageMakerand encouraged you to explore it further!


# Additional Resources

There is a lot more on multi-models endpoints to cover!

I recommend to read the following site carefully: [link](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html)

Specially the sections on [how multi-model works](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html#multi-model-endpoint-instance), [model caching](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html#multi-model-endpoint-instance) and [instance recommendations](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html#multi-model-endpoint-instance). 

On top of that, the multi-model server documentation is also a great resource:
- [multi-model server repo](https://github.com/awslabs/multi-model-server)
- [Inference API](https://github.com/awslabs/multi-model-server/blob/master/docs/inference_api.md)
- [Management API](https://github.com/awslabs/multi-model-server/blob/master/docs/management_api.md)
- [multi-model server docs](https://github.com/awslabs/multi-model-server/tree/master/docs)

