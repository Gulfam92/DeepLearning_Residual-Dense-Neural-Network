# Abstract
"""Deeper neural networks are difficult to train and pose vanishing gradients problems
while training the network. To overcome these challenges, various neural network 
architectures have been proposed in recent times and these newly proposed architectures
have helped researchers in the deep learning area in the image classification category by
improving the accuracy rate. Resnet and Densenet are few such neural net architectures
which have helped the training of networks that are substantially deeper than those used
previously. Neural networks with multiple layers in the range of 100-1000 dominates
image recognition tasks, but building a network by simply stacking residual blocks
limits its optimization problem as pointed in the ResNet paper and the authors have
shown that architectures with thousands layers do not add much and performs similar to
architecture with hundred layers or even worse than that. This paper proposes a network
architecture Residual Dense Neural Network (ResDen), to dig the optimization ability of
neural networks. With enhanced modeling of Resnet and Densenet, this architecture is
easier to interpret and less prone to overfitting than traditional fully connected layers or
even architectures such as Resnet with higher levels of layers in the network. Our
experiments demonstrate the effectiveness in comparison to ResNet models and achieve
better results on CIFAR-10 dataset."""


***Steps to run python file:***<br>
<br>
1. To run, execute command "python train.py --depth 52 --schedule 120 200" or "CUDA_VISIBLE_DEVICES=0 python train.py --depth 52 --schedule 120 200".<br>
2. The dataset should be stored in folder CIFAR which will be downloaded automatically while executing the above command.<br>
3. Change Path accordingly in 'train.py'.<br>
