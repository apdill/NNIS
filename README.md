# Neural Network (Instance) Segmentation Library

This library provides tools for generating, processing, and segmenting synthetic images of neurons and their dendritic arbors, as well as using PyTorch for neural network-based instance segmentation. The core focus is on training neural networks to distinguish between individual neurons and their dendrites in fluorescence microscopy images.

## Features

- **Synthetic Image Generation**: 
  - Generates synthetic 2D neuron network images and corresponding segmentation masks using fractal-based neuron models.
  - Supports customizable neuron properties, including soma size, dendrite branching depth, and branch angles.

- **Instance Segmentation**: 
  - Trainable neural network models for distinguishing individual neurons and their dendrites.
  - Supports binary segmentation and multi-class segmentation.
  - Post-processing tools such as connected component analysis and watershed segmentation for refining predictions.

- **Data Handling**:
  - Functions to read and process neuron images, masks, and DataFrames.
  - Efficient handling of TIFF images for large-scale training datasets.
  - Tools for loading, augmenting, and batching data for neural network training.
