# MRDC-CNN


# Paper
"A Deep Learning Method for High-Quality Ultra-Fast CT Image Reconstruction from Sparsely Sampled Projections"
In revision process in Nuclear Inst. and Methods in Physics Research, A (No.: NIMA-D-21-00934).

# Abstract 
Few-view or sparse-view computed tomography has been recently introduced as a great potential to speed up data acquisition and alleviate the amount of patient radiation dose. This study aims to present a method for high-quality ultra-fast image reconstruction from sparsely sampled projections to overcome problems of previous methods, missing and blurring tissue boundaries, low-contrast objects, variations in shape and texture between the images of different individuals, and their outcomes. To this end, a new deep learning (DL) framework based on convolution neural network (CNN) models is proposed to solve the problem of CT reconstruction under sparsely sampled data, named the multi-receptive field densely connected CNN (MRDC-CNN). MRDC-CNN benefits from an encoder-decoder structure by proposing dense skip connections to recover the missing information, multi-receptive field modules to enlarge the receptive field, and having no batch normalization layers to boost the performance. The MRDC-CNN with a hybrid loss function format introduces several auxiliary losses combined with the main loss to accelerate convergence rate and alleviate the gradient vanishing problem during network training and maximize its performance. Results have shown that MRDC-CNN is 4-6 times faster than the state-of-the-art methods, with fewer memory requirements, better performance in other objective quality evaluations, and improved visual quality. The results indicated the superiority of our proposed method compared to the latest algorithms. In conclusion, the proposed method could lead to high quality CT imaging with quicker imaging speed and lower radiation dose.

# Algorithm
The flowchart illustration of MRDC-CNN for sparse-view CT reconstruction is as follows:



# Usage
The repository contains code to train the proposed reconstruction method (MRDC-CNN). Implementation of MRDC-CNN is performed on the Keras library, running on the machine learning platform TensorFlow.

1. The data is not contained in this repository and needs to be uploaded.
Check (and modify if necessary) the directory paths for the data and results. By default, the data should be stored in the subdirectory Data and models are stored in the subdirectory models.
2. Train networks using the script named *Train.py*.
3. You can evaluate the trained networks on the test data using the script named *Test.ipynb*.

# Result

Figures 1-3 portray representative CT reconstruction results of the MRDC-CNN for various organs as an example.

![Colon](https://user-images.githubusercontent.com/42764887/149763656-02e82e6a-7ca0-4284-94f5-b694d3387c1a.png)
Fig. 1. Representative reconstructed qualitative results from different views on a representative trunk-test image of the colon region. The reference images are presented in the first and second columns of the first row. Columns 3-5 of the first row contained reconstructed images of FBP (used as the input of our network), and the second row shows reconstructed results of the MRDC-CNN.

![Kidney](https://user-images.githubusercontent.com/42764887/149764076-f188734e-0d4c-4dee-a614-ccd0abf6612c.png)
Fig. 2. Representative reconstructed qualitative results from different views on a representative trunk-test image of the kidney region. The reference images are presented in the first and second columns of the first row. Columns 3-5 of the first row contained reconstructed images of FBP (used as the input of our network), and the second row shows reconstructed results of the MRDC-CNN.


![Liver_Tumor](https://user-images.githubusercontent.com/42764887/149764216-11523a8d-2ce7-4f4d-a4bd-59a9f6f6c86f.png)
Fig. 3. Representative reconstructed qualitative results from different views on a representative trunk-test image of the liver region. The reference images are presented in the first and second columns of the first row. Columns 3-5 of the first row contained reconstructed images of FBP (used as the input of our network), and the second row shows reconstructed results of the MRDC-CNN.
