---
layout: post
permalink: /01ead88355127f33e70117da2578556e80ac1dbc/:categories/:year/:month/:day/:title:output_ext
title:  "SageMaker Asynchronous Endpoint Auto Scaling to Zero"
description: "Exploring cost savings by dynamically scaling endpoints based on demand"
type: card-dated
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
categories: post
tags: 
  - AWS
  - Sagemaker
author: Jose
card: card-1
---


# SageMaker Asynchronous Endpoint Auto Scaling to Zero

## Introduction

In this post, I explain my experience testing the horizontal auto scaling feature for Amazon SageMaker asynchronous endpoints. 

## What is Scaling?

Scaling in AWS refers to adjusting capacity of services to handle changes in demand. There are two main types:

- **Vertical Scaling:** Increasing capacity of an existing resource by adding more CPU, memory, storage, etc. 
- **Horizontal Scaling:** Adding or removing resources to increase or decrease the number of instances handling the workload.

AWS offers automatic scaling for various services like EC2, RDS, and DynamoDB to adjust capacity in response to demand changes. This ensures optimal performance and cost-efficiency.

## Target of This Test

The target is horizontal autoscaling. Autoscaling automatically adjusts the number of resources based on changing demand. It prevents overprovisioning and underprovisioning. 

The test endpoint is a SageMaker asynchronous endpoint running a single Pytorch Donut model. 

The `Donut` model was proposed in OCR-free Document Understanding Transformer by Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park. 

Donut consists of an image Transformer encoder and an autoregressive text Transformer decoder to perform document understanding tasks such as document image classification, form understanding and visual question answering.

## Scaling to Zero

Scaling to zero means dynamically scaling down to zero instances when there is no demand. This shuts down resources, freeing them up and reducing costs for infrequent or irregular workloads.

The goal is to analyze how scaling to zero performs for this SageMaker endpoint.

## Configuration

Using the Python SDK, I created a policy with:

- Max capacity: 1 instance 
- Min capacity: 0 instances
- Scale in and out cooldown periods
- Target metric: SageMakerVariantInvocationsPerInstance

This allows scaling between 0 and 1 instance based on the target metric.

Python code:

```
client = boto3.client("application-autoscaling") 
# Common class representing Application Auto Scaling for SageMaker amongst other services
resource_id = ("endpoint/" + endpoint_name + "/variant/" + "variant1")  
# This is the format in which application autoscaling references the endpoint

# Configure Autoscaling on asynchronous endpoint down to zero instances
response = client.register_scalable_target(
    ServiceNamespace="sagemaker", 
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=0,
    MaxCapacity=1,
)

response = client.put_scaling_policy(
    PolicyName="Invocations-ScalingPolicy",
    ServiceNamespace="sagemaker",  
    # The namespace of the AWS service that provides the resource.
    ResourceId=resource_id,  
    # Endpoint name
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",  
    # SageMaker supports only Instance Count
    PolicyType="TargetTrackingScaling",  
    # 'StepScaling'|'TargetTrackingScaling'
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 1.0,  
        # The target value for the metric. - here the metric is - SageMakerVariantInvocationsPerInstance
        "CustomizedMetricSpecification": {
            "MetricName": "ApproximateBacklogSizePerInstance",
            "Namespace": "AWS/SageMaker",
            "Dimensions": [{
                "Name": "EndpointName",
                "Value": endpoint_name
            }],
            "Statistic": "Average", 
        },
        "ScaleInCooldown": 150,  
        # The cooldown period helps you prevent your Auto Scaling group from launching or terminating
        # additional instances before the effects of previous activities are visible.
        # You can configure the length of time based on your instance startup time or other application needs.
        # ScaleInCooldown - The amount of time, in seconds, after a scale in activity completes before another scale in activity can start.
        "ScaleOutCooldown": 300  
        # ScaleOutCooldown - The amount of time, in seconds, after a scale out activity completes before another scale out activity can start.
        # 'DisableScaleIn': True|False - ndicates whether scale in by the target tracking policy is disabled. 
        # If the value is true, scale in is disabled and the target tracking policy won't remove capacity from the scalable resource.
    },
)
```

## Results

After 300 seconds of inactivity, the endpoint scaled to 0 instances, costing nothing. 

Sending 5 requests triggered the scale in policy. The endpoint updated and took ~10 minutes to start a new instance. Once ready, it processed the 5 requests sequentially as normal.

The first request took ~11 minutes total processing time. The pending request queue spiked from 0 to 5 and back to 0.

## Conclusion

Scaling to zero can provide cost savings for workloads with irregular demand patterns. However, scale in time was very long at 10 minutes in this test. Using lighter Docker images or models could potentially improve this. 

Overall, effectiveness depends heavily on specific project requirements.
