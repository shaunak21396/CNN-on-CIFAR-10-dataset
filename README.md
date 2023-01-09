# CNN-on-CIFAR-10-dataset
Image Classification task using various models (from a simple Softmax classifier to a Residual Network) on CIFAR-10 dataset images.
It contains 60,000 32x32 RGB images of 10 different classes (50,000 for training and 10,000 for testing)

Reference: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

<h2>What I learned:</h2>

- <b>Application of the following Python libraries/packages:
  1. Numpy
  2. Matplotlib
  3. Pytorch
  4. OpenCV
  5. Imageio

- <b>Implementation of various models for Image Classification Task like:
  - <b>Softmax Classifier(shallow network)
  - <b>Softamx Classtifier with Adam Optimizer
  - <b>MLP using tanh activation function
  - <b>MLP using tanh activation function on Augmented data
  - <b>MLP using relu activation function on Augmented data
  - <b>Convolutional Neural Network with convoluion, pooling, relu and linear layers on Augmented data
  - <b>Convolutional Neural Network with convoluion, pooling, relu and linear layers on Augmented data with Batch Normalization 
  - <b>Convolutional Neural Network with Strided Convolutions
  - <b>Convolutional Neural Network with Global Pooling
  - <b>Residual Network
- <b>Comparison between all the above mentioned models on the basis of training loss, training accuracy, test loss, test accuracyand time for each epoch.
- <b>Implementation of Optimizers like Adam improves the accuracy.
- <b>Data Augmentation virtually expands the dataset such that model cannot memorize the training samples that easily (avoids overfitting)
- <b>Number of epocs play a critical role in the training process. Here we observe that Training curves improve during training, testing curves worsen after ~20 epochs.
- <b>Implementation of Activation functions also improves the ytest accuracy (Here Relu performs better than tanh)
- <b>CNN have significantly better accuracy but at the cost of training time.
- <b>Initialization of the weights plays a critical role in the learning process. Hence proper initialization (here He-initialization) is taken into consideration.
- <b>Training is faster with strided convolution.
- <b>Residual Networks have the highest accuracy but training time was large.
