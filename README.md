# Foot Distance Measurement

This project is designed to measure the distance between two points on a foot using image processing techniques.

## Problem Definition

Imagine two parallel lines receding into the distance.  In 3D space, they remain parallel, but in the 2D image, they appear to converge.  This is perspective distortion.  It affects both size and relative position.  The further away an object is, the smaller it appears, and the closer together its projected features seem.

![Problem Illustration](assets/img/problem.jpeg)

## Why Simple 2D Distance Fails

- The formula that we use to calculate the is 
    ```distance = √((x2 - x1)² + (y2 - y1)²)```, calculates the distance in pixel space.  

- It doesn't account for perspective.  

- As the person moves away, the pixel distance between their ankles shrinks, even if the real-world distance stays the same.

## Introduction

The Foot Distance Measurement project aims to provide an accurate and efficient way to measure distances on a foot from images. This can be useful for medical purposes, shoe fitting, and other applications.

## Different approach to takle the depth information

1. Baseline approach :  
    - The problem of magnitude of the distance between the feet is not accurate

2. Approximate Depth Estimation (Simplest, but Least Accurate):
    - If we have any information about the person's height or some other known dimension in the scene.
    - we can use that to estimate a scaling factor related to depth. 
    - For example, if we know the average height of a person, and you measure the person's height in the image we can get a rough estimate of how much the image is "scaled down" due to distance.
    - So, we get an information of the scale that affected by the distance.
    - So, we can use it by multipling the calculated 2D distance by this scaling factor. 
    - This will give us a distance that's less sensitive to changes in depth, but it's still an approximation. 

3. Depth estimation using Z value from mediapipe 
    - The idea is to get the (x,y,z) value from mediapipe pose estimation module 

4. Depth estimation using MiDas model 

5. Depth estimation using depth anything model with different encoder variations 

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

# How to use 

run the command 

```
python main.py
```
