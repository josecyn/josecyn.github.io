---
layout: post
permalink: /01ead88355127f33e70117da2578556e80ac1dbc/:categories/:year/:month/:day/:title:output_ext
title:  "Testing the Sagemaker Asynchronous Endpoint"
description: "Exploring the Sagemaker Asynchronous Endpoint"
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


# Sagemaker Asynchronous Endpoint Test

## Introduction

In this document I explain my findings regarding Sagemaker asynchronous endpoint.

## Architecture 

The basic architecture is as follows:

![Architecture](/assets/img/posts/sagemaker_test/architecture.png)

### Workflow:

1. The user invokes the async endpoint. He makes n request(s).
2. The async endpoint manages the requests in an internal queue (handled by Sagemaker)
3. The async endpoint processes the requests.
4. The async endpoint stores the results in the specified S3 bucket. 
5. (Optional) The async endpoint notifies success and errors to specific SNS topics.

## Deploy Model

Given a trained model, deploying a model is very simple. 

You need:

- docker image. 
- model.tar.gz : tar.gz file that contains the model and inference code.

### Docker Image

I searched a suitable Docker Sagemaker image here: [available images](https://github.com/aws/deep-learning-containers/blob/master/available_images.md)

One that suits this test was: `763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker`

Properties:

- SageMaker Framework Container (important to reduce the source code you have to write)
- PyTorch version 1.11.0 
- python version 3.8 (py38)
- inference (not training)
- No Horovod
- GPU available

### Prepare model.tar.gz

I did this step manually, outside the Sagemaker environment. 
Following the information here. 
I created a `tar.gz` file with the following structure:

```
|-- my_model
    |-- model.pth

|-- code
    |-- inference.py 
    |-- requirements.txt
```

#### File model.pth

The Pytorch trained model. Just add it to the tar.gz. The model is a `donut` model with image input size 960x640.

The `Donut` model was proposed in OCR-free Document Understanding Transformer by Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park. 

Donut consists of an image Transformer encoder and an autoregressive text Transformer decoder to perform document understanding tasks such as document image classification, form understanding and visual question answering.

#### Code Folder 

The code folder is a bit more tricky. You have to comply with the Sagemaker requeriments.

But in the case of donut it was fairly easy. 

##### File requirements.txt

requirements.txt includes all the python dependencies to run your code. 
In my case, I had to add the following packages:

```
pytorch-lightning==1.6.4
transformers==4.11.3
timm==0.5.4 
donut-python==1.0.9
```

##### Inference Code

In the inference code you have to do the standard tasks:

- Load the model
- Preprocess input 
- Execute inference
- Postprocess 
- Return predictions

You are free to return whatever output you want. 
In this case, given an image as input, I return raw JSON with the predictions.

Information below is based on the following source.

The inference.py file contains your custom inference module, and the requirements.txt file contains additional dependencies that should be added. The custom module can override the following methods:

- `model_fn(model_dir)` overrides the default method for loading a model. The return value `model` will be used in `predict` for predictions. `predict` receives argument the `model_dir`, the path to your unzipped `model.tar.gz`.
- `input_fn(input_data, content_type)` overrides the default method for preprocessing. The return value `data` will be used in `predict` for predicitions. The inputs are:
  - `input_data` is the raw body of your request.
  - `content_type` is the content type from the request header.
- `predict_fn(processed_data, model)` overrides the default method for predictions. The return value `predictions` will be used in `postprocess`. The input is `processed_data`, the result from `preprocess`. 
- `output_fn(prediction, accept)` overrides the default method for postprocessing. The return value `result` will be the response of your request (e.g.JSON). The inputs are:
  - `predictions` is the result from `predict`.
  - `accept` is the return accept type from the HTTP Request, e.g. `application/json`.

Here is the skeleton of a custom inference module with `model_fn`, `input_fn`, `predict_fn`, and `output_fn`:

```python
def model_fn(model_dir):
  return "model"

def input_fn(data, content_type):
  return "data"

def predict_fn(data, model):
  return "output"  

def output_fn(prediction, accept):
  return "prediction"
```

Here's an implementation example: [link](https://github.com/aws/sagemaker-inference-toolkit/blob/master/src/sagemaker_inference/default_inference_handler.py). 

And here's another one more complete, it also includes training: [link](https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/sklearn/estimator.py#L309-L504).


#### Environment Variable

The last thing is telling the Sagemaker that the entry point is the file inference.py. 
You do that with a Environment Variable at deploy time.

## Create Sagemaker Model 

Finally, using the AWS web console, we put together the previous steps to create the Sagemaker model. 
You have to enter:

- Sagemaker execution role
- model file S3 location (the model.tar.gz we prepared). 
- Docker Image.
- The env variable key-value pair.

## Create Sagemaker Endpoint Config

Next, you have to create the endpoint configuration. I completed this step using the AWS web console. Elements:

- Type of endpoint: provisioned 
- S3 output path. This is optional. If not given, the output will be included in the success notification.
- (Optional) Success/Error notification location: SNS topic/SQS queue. 
- Variants

In addition, you have to specify "variants" for your endpoint configuration. 
Here, you tell the Sagemaker which model to use, what type of instance, how many etc.

Note: Async endpoints do not support Amazon Inferentia chips.

## Create Sagemaker Enpoint

Finally, with the model and the enpoint configuration ready, you can create the endpoint.
You just basically select the previously created endpoint configuration. 

## Test Endpoint

If you successfully completed the steps above, you can now send requests to your endpoint using the method `invoke_endpoint_async`. 

Python example:

```python
# Create a low-level client representing Amazon SageMaker Runtime
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name='ap-northeast-1')

# The name of the endpoint. The name must be unique within an AWS Region in your AWS account  
endpoint_name='donut-m-async-endpoint'

# Specify the location of the input. Here, a single SVM sample
input_location = "s3://chek-project/received/asdj48927r3rn.jpg"

# After you deploy a model into production using SageMaker hosting services, your client applications use this API to get inferences from the model hosted at the specified endpoint.
response = sagemaker_runtime.invoke_endpoint_async(
                            EndpointName=endpoint_name,
                            InputLocation=input_location,
                            ContentType='application/x-image')
```

Here you can find the details of the function `invoke_endpoint_async` [link](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html#SageMakerRuntime.Client.invoke_endpoint_async).

Remember to set your region, endpoint name, and input location. 
Input location is an S3 path.

I ran the following code pushing 100 images sequentially. 

The Sagemaker async endpoint processed them one by one. In total it took 28 seconds. 
Which means that each image took in avg 0.28 seconds. In line with our previous tests using executing inference of 1 single image on the same instance type `ml.g4dn.xlarge`.

Using the instance type ml.m5d.xlarge inference took <= 5 seconds.

Here's a python script to test endpoint before deployment: [link](https://github.com/aws-samples/amazon-sagemaker-endpoint-tests/blob/main/test-async-endpoint/src/integration-test/invoke_endpoint.py)

### Response 

The client invokes the async endpoint passing the S3 location of a file.
The response that the client gets is as follows:

```
{
    'ResponseMetadata': { 
        'RequestId': '08886034-9ca5-4604-a2a7-cd1c69a807e1', 
        'HTTPStatusCode': 202, 
        'HTTPHeaders': { 
            'x-amzn-requestid': '08886034-9ca5-4604-a2a7-cd1c69a807e1', 
            'x-amzn-sagemaker-outputlocation': 's3://bucket/async_output/61fe991.out', 
            'date': 'Tue, 31 Jan 2023 00:28:16 GMT', 
            'content-type': 'application/json',
            'content-length': '54'
        }, 
        'RetryAttempts': 0
    }, 
    'OutputLocation': 's3://bucket/async_output/61fe991.out', 
    'InferenceId': '115-2-4-adc-b5592c'
}
```

# Conclusions

- Very simple deployment process
- Developer just focuses on writing the inference code
- Robust asynchronous architecture
- Endpoint is running 24x7 → costs could be high depending on selected instance type
- Scalability available (it can scale down to 0, but start time is very slow).