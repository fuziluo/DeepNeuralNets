# DeepNeuralNets
A deep learning framework written in Java with GPU acceleration (OpenCL).

# Features
* Can be used with any GPU which supports OpenCL(1.1). However code has only been tuned in AMD and apple platform. Nvidia and Intel has not been thoroughly tested but not major problem forseen.
* As it's in Java. Basically there is no OS limitations.
* Support convolutional layer, pooling layer and fully-connected layer.
* Support activation types include Sigmoid, Tanh, Relu and Softmax.
* The training uses stochastic gradient descent with available parameters such as momentum, weight decay.
* Weight initialisation with Gaussian distribution and uniform distribution.

# Test
The framework was tested using MNIST and CIFAR-10 with Caffe as benchmark (PR2610 https://github.com/BVLC/caffe/pull/2610).
With the same dataset, model, parameters, OS and hardware, this framework produces the same classification accuracy in much shorter training time. For MNIST it's 2.4 times faster and for CIFAR 2.6.
