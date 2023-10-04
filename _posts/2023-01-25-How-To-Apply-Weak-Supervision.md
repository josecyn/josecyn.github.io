---
layout: post
permalink: /01ead88355127f33e70117da2578556e80ac1dbc/:categories/:year/:month/:day/:title:output_ext
title:  "How to Apply Weak Supervision"
description: "Weak supervision description"
type: card-dated
image: http://placehold.it/750X300?text=Header+Image # for local images, place in /assets/img/posts/
caption: 
categories: post
tags: 
  - Weak Supervision
author: Jose
card: card-1
---

# Applying Weak Supervision

The purpose of this post is to explain how to apply weak supervision with a practical example.

## The Situation

After a successful proof of concept, we had an initial model that could detect seals and organization blocks in invoices with relatively good accuracy. But the model was trained with only around 300 images. As we had much more unlabeled data available, we could create a more robust model. 

The problem? The additional data was not labeled. Labeling another 2,700 images would be expensive, so we decided to apply weak supervision to increase the number of labeled images in the dataset.

## How Weak Supervision Works

The idea behind weak supervision is to use an existing model trained with a small labeled dataset to generate labels for a larger unlabeled dataset. 

Of course, the model will have limited accuracy and make labeling mistakes. At that point, you have two options with the newly labeled data:

1. Correct the generated labels
2. Retrain the model without corrections

Whether to correct labels depends on the model's accuracy and the expense of manual corrections. This is why it's called weak supervision - the model provides supervision, but it is noisy and imprecise.

According to Wikipedia, weak supervision is:

> A branch of machine learning where noisy, limited, or imprecise sources are used to provide supervision signal for labeling large amounts of training data in a supervised learning setting.

Corrected labels will provide better data to learn from. However, sometimes the model can still improve without correcting labels.

## Applying Weak Supervision

Here's a summary of how to apply weak supervision:

1. **Label some data** - Depending on the task complexity and labeling costs, start with a small labeled dataset. Harder problems need more data. 

2. **Train your first model** - Train a model on the labeled data. If it doesn't learn well, increase the training set. Validation and testing are less important at this stage.

3. **Evaluate model accuracy** - Assess model accuracy, especially if this is the second or later iteration. If it's too low, continue to the next steps.

4. **Generate new labels** - Use the trained model to label more unlabeled data, ideally 2-3x more. Model accuracy determines how much data it can reliably label.

5. **Correct labels (optional)** - Decide if it's worthwhile to manually correct some percentage of the newly generated labels. This improves model accuracy but takes time. 

6. **Retrain the model** - Retrain the model on the new labeled dataset. Repeat the process by evaluating again and generating more labels.

The key steps are:

- Manually label some data 
- Train a model
- Evaluate accuracy
- Use the model to generate labels for more data
- Optionally correct some labels
- Iterate - retrain on new data

By repeating this process, weak supervision leverages models to rapidly label large datasets for improved accuracy.

There is an obvious problem: it can reinforce errors and even diverge. 2 possible fixes could be:

- Only use labels the model is very confident about
- Regularize the loss from the unlabeled examples

I hope this explanation helps provide a clear overview of how to apply weak supervision to machine learning tasks. 

# Links

* [CPSC 340: Machine Learning and Data Mining - Semi-Supervised Learning (Fall 2019)](https://www.cs.ubc.ca/~schmidtm/Courses/LecturesOnML/semiSupervised.pdf)