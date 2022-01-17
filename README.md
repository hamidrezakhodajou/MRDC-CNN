# MRDC-CNN


# Paper
"A Deep Learning Method for High-Quality Ultra-Fast CT Image Reconstruction from Sparsely Sampled Projections"
In revision process in Nuclear Inst. and Methods in Physics Research, A (No.: NIMA-D-21-00934).

# Abstract 
Few-view or sparse-view computed tomography has been recently introduced as a great potential to speed up data acquisition and alleviate the amount of patient radiation dose. This study aims to present a method for high-quality ultra-fast image reconstruction from sparsely sampled projections to overcome problems of previous methods, missing and blurring tissue boundaries, low-contrast objects, variations in shape and texture between the images of different individuals, and their outcomes. To this end, a new deep learning (DL) framework based on convolution neural network (CNN) models is proposed to solve the problem of CT reconstruction under sparsely sampled data, named the multi-receptive field densely connected CNN (MRDC-CNN). MRDC-CNN benefits from an encoder-decoder structure by proposing dense skip connections to recover the missing information, multi-receptive field modules to enlarge the receptive field, and having no batch normalization layers to boost the performance. The MRDC-CNN with a hybrid loss function format introduces several auxiliary losses combined with the main loss to accelerate convergence rate and alleviate the gradient vanishing problem during network training and maximize its performance. Results have shown that MRDC-CNN is 4-6 times faster than the state-of-the-art methods, with fewer memory requirements, better performance in other objective quality evaluations, and improved visual quality. The results indicated the superiority of our proposed method compared to the latest algorithms. In conclusion, the proposed method could lead to high quality CT imaging with quicker imaging speed and lower radiation dose.

~~# Implementation~~
~~MatConvNet (matconvnet-1.0-beta24)~~
~~Please run the matconvnet-1.0-beta24/matlab/vl_compilenn.m file to compile matconvnet.
~~There is instruction on http://www.vlfeat.org/matconvnet/mfiles/vl_compilenn/~~
~~Frameing U-Net (matconvnet-1.0-beta24/examples/framing_u-net)
~~Please run the matconvnet-1.0-beta24/examples/framing_u-net/install.m~~
~~Install the customized library
~~Download the trained networks such as standard cnn, u-net, and tight-frame u-net~~

# Source code and Implementation

Implementation of MRDC-CNN is performed on TensorFlow framework with Python language. Source code to train is *Train.py* in companion with *Network.py* as a subroutine contaning the structure of MRDC-CNN. The source code used ML libraries of both Keras and PyTorch.


# Test data

Figures 1-3 portray representative CT reconstruction results of the MRDC-CNN by using *Test.ipynb* for various organs as an example.

![Colon](https://user-images.githubusercontent.com/42764887/149763656-02e82e6a-7ca0-4284-94f5-b694d3387c1a.png)
Fig. 1. Representative reconstructed qualitative results from different views on a representative trunk-test image of the colon region. The reference images are presented in the first and second columns of the first row. Columns 3-5 of the first row contained reconstructed images of FBP (used as the input of our network), and the second row shows reconstructed results of the MRDC-CNN.

![Kidney](https://user-images.githubusercontent.com/42764887/149764076-f188734e-0d4c-4dee-a614-ccd0abf6612c.png)
Fig. 2. Representative reconstructed qualitative results from different views on a representative trunk-test image of the kidney region. The reference images are presented in the first and second columns of the first row. Columns 3-5 of the first row contained reconstructed images of FBP (used as the input of our network), and the second row shows reconstructed results of the MRDC-CNN.


![Liver_Tumor](https://user-images.githubusercontent.com/42764887/149764216-11523a8d-2ce7-4f4d-a4bd-59a9f6f6c86f.png)
Fig. 3. Representative reconstructed qualitative results from different views on a representative trunk-test image of the liver region. The reference images are presented in the first and second columns of the first row. Columns 3-5 of the first row contained reconstructed images of FBP (used as the input of our network), and the second row shows reconstructed results of the MRDC-CNN.
