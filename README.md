# Image Classification with CIFAR-10 Dataset

In this project, we will train various machine learning models to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

## Methods

### Data Loading and Preprocessing

We will use a preferred machine learning library to load and preprocess the CIFAR-10 dataset. The dataset is divided into training and testing sets using the library's API. For the purposes of this problem, we will randomly sample 20% of the training set and use it as a new training set, while the original test set will be used for validation.

### Multi-Layer Perceptron (MLP)

We will build a multi-layer perceptron (MLP) for image classification. The MLP architecture includes:
- Fully connected layer with 512 units and a sigmoid activation function.
- Fully connected layer with 512 units and a sigmoid activation function.
- Output layer with the suitable activation function and number of neurons for the classification task.

### Convolutional Neural Network 1 (CNN1)

We will build a convolutional neural network (CNN) for image classification. The CNN architecture includes:
- 2D Convolutional layer with 64 filters (size of 3x3) and ReLU activation function.
- 2D Convolutional layer with 64 filters (size of 3x3) and ReLU activation function.
- Fully connected (Dense) layer with 512 units and a sigmoid activation function.
- Fully connected layer with 512 units and a sigmoid activation function.
- Output layer with the suitable activation function and number of neurons for the classification task.

### Convolutional Neural Network 2 (CNN2)

We will also build another convolutional neural network (CNN) for image classification. The CNN architecture includes:
- 2D Convolutional layer with 64 filters (size of 3x3) and ReLU activation function.
- 2x2 Max pooling layer.
- 2D Convolutional layer with 64 filters (size of 3x3) and ReLU activation function.
- 2x2 Max pooling layer.
- Fully connected layer with 512 units and a sigmoid activation function.
- Dropout layer with 0.2 dropout rate.
- Fully connected layer with 512 units and a sigmoid activation function.
- Dropout layer with 0.2 dropout rate.
- Output layer with the suitable activation function and number of neurons for the classification task.

### Training and Evaluation

We will use a batch size of 32 for training both MLP and CNN models. The Adam optimizer will be used to optimize the models' parameters. The choice of loss function will be appropriate for multi-class classification, and the accuracy metric will be monitored during training.

Each network will be trained for 5 epochs to ensure convergence and to capture the learning trends.

This project aims to demonstrate the implementation and training of various models for image classification on the CIFAR-10 dataset. By evaluating the accuracy of these models on the validation set, we can compare their performance and gain insights into their effectiveness for this task.
