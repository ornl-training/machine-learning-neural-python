---
title: Neural networks
teaching: 40
exercises: 20
---

::::::::::::::::::::::::::::::::::::::: objectives

- Understand the structure and components of a neural network.
- Identify the purpose of activation functions and dense layers.
- Explain how convolutional layers extract features from images.
- Construct a convolutional neural network using PyTorch.

::::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::::: questions

- What is a neural network and how is it structured?
- What role do activation functions play in learning?
- What is the difference between dense and convolutional layers?
- Why are convolutional neural networks effective for image classification?

::::::::::::::::::::::::::::::::::::::::::::::::::


## What is a neural network?

An artificial neural network, or just "neural network", is a broad term that describes a family of machine learning models that are (very!) loosely based on the neural circuits found in biology.

The smallest building block of a neural network is a single neuron. A typical neuron receives inputs (x1, x2, x3) which are multiplied by learnable weights (w1, w2, w3), then summed with a bias term (b). An activation function (f) determines the neuron output.

![](fig/neuron.png){alt='Neuron' width="600px"}

From a high level, a neural network is a system that takes input values in an "input layer", processes these values with a collection of functions in one or more "hidden layers", and then generates an output such as a prediction. The network has parameters that are systematically tweaked to allow pattern recognition.

![](fig/simple_neural_network.png){alt='Neuron' width="800px"}

The layers shown in the network above are "dense" or "fully connected". Each neuron is connected to all neurons in the preceeding layer. Dense layers are a common building block in neural network architectures.

"Deep learning" is an increasingly popular term used to describe certain types of neural network. When people talk about deep learning they are typically referring to more complex network designs, often with a large number of hidden layers.

## Activation Functions

Part of the concept of a neural network is that each neuron can either be 'active' or 'inactive'. This notion of activity and inactivity is attempted to be replicated by so called activation functions. The original activation function was the sigmoid function (related to its use in logistic regression). This would make each neuron's activation some number between 0 and 1, with the idea that 0 was 'inactive' and 1 was 'active'.

As time went on, different activation functions were used. For example the tanh function (hyperbolic tangent function), where the idea is a neuron can be active in both a positive capacity (close to 1), a negative capacity (close to -1) or can be inactive (close to 0).

The problem with both of these is that they suffered from a problem called [model saturation](https://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_SSCI_2015/data/7560b423.pdf). This is where very high or very low values are put into the activation function, where the gradient of the line is almost flat. This leads to very slow learning rates (it can take a long time to train models with these activation functions).

One popular activation function that tries to tackle this is the rectified linear unit (ReLU) function. This has 0 if the input is negative (inactive) and just gives back the input if it is positive (a measure of how active it is - the metaphor gets rather stretched here). This is much faster at training and gives very good performance, but still suffers model saturation on the negative side. Researchers have tried to get round this with functions like 'leaky' ReLU, where instead of returning 0, negative inputs are multiplied by a very small number.

![](fig/ActivationFunctions.png){alt='Activation functions' width="600px"}

## Convolutional neural networks

Convolutional neural networks (CNNs) are a type of neural network that especially popular for vision tasks such as image recognition. CNNs are very similar to ordinary neural networks, but they have characteristics that make them well suited to image processing.

Just like other neural networks, a CNN typically consists of an input layer, hidden layers and an output layer. The layers of "neurons" have learnable weights and biases, just like other networks.

What makes CNNs special? The name stems from the fact that the architecture includes one or more convolutional layers. These layers apply a mathematical operation called a "convolution" to extract features from arrays such as images.

In a convolutional layer, a matrix of values referred to as a "filter" or "kernel" slides across the input matrix (in our case, an image). As it slides, values are multiplied to generate a new set of values referred to as a "feature map" or "activation map".

![](fig/convolution_plotke.gif){alt='2D Convolution Animation by Michael Plotke' width="500px"}

Filters provide a mechanism for emphasising aspects of an input image. For example, a filter may emphasise object edges. See [setosa.io](https://setosa.io/ev/image-kernels/) for a visual demonstration of the effect of different filters.

## Max pooling

Convolutional layers often produce large feature maps — one for each filter. To reduce the size of these maps while retaining the most important features, we use **pooling**.

The most common type is **max pooling**. It works by sliding a small window (often 2×2) across the feature map and taking the **maximum value** in each region. This reduces the resolution of the feature map (for a 2x2 window by a factor of 2) while keeping the strongest responses.

For example, if we apply max pooling to the following 4×4 matrix:

```
[1, 3, 2, 1],
[5, 6, 1, 2],
[4, 2, 9, 8],
[3, 1, 2, 0]
```

We get this 2×2 output:

```
[6, 2],
[4, 9]
```

Each value in the output is the maximum from a 2×2 window in the input.

### Why use max pooling?

- **Reduces computation** by shrinking the feature maps
- **Adds translation tolerance** — the model is less sensitive to small shifts in the image
- **Keeps the strongest features** while discarding low-importance details

In PyTorch, max pooling is implemented with the `nn.MaxPool2d()` layer. You'll see it applied multiple times in our network to gradually reduce the size of the feature maps and focus on the most prominent features.

## Dropout

When training neural networks, a common problem is overfitting — the model learns to perform very well on the training data but fails to generalize to new, unseen examples.

Dropout is a regularization technique that helps reduce overfitting. During training, dropout temporarily "drops out" (sets to zero) a random subset of neurons in a layer. This forces the network to learn redundant representations and prevents it from becoming too reliant on any single path through the network.

In practice:

- During training: a random set of neurons is deactivated at each step.
- During inference (prediction), all neurons are used, and the outputs are scaled accordingly.

For example:

```python
from torch import nn

x = nn.Linear(128) # Example dense layer
# Drop 50% of neurons during training
x = nn.Dropout(0.5)(x)
```

The value 0.5 is the dropout rate — the fraction of neurons to disable.

::::::::::::::::::::::::::::::::::::::: challenge

A) Why is dropout helpful during training?
B) What effect do you expect from reducing or removing the dropout rate during training?

::::::::::::::: solution

A) Dropout randomly disables neurons during training, forcing the network to not rely too heavily on any one path. This helps prevent overfitting and improves generalization.

B) With lower or no dropout, training accuracy may rise faster, but validation accuracy may stagnate or decline, indicating overfitting.

:::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::

## Creating a convolutional neural network

Before training a convolutional neural network, we will first define its architecture. The architecture we use in this lesson is intentionally simple. It follows common CNN design principles:

- Repeated use of small convolutional filters (3×3 or 5×5)
- Max pooling to reduce dimensionality
- Fully connected layers at the end for classification

This architecture is loosely inspired by classic CNNs such as LeNet-5 and VGGNet. It strikes a balance between performance and clarity. It’s small enough to train on a CPU, but expressive enough to learn meaningful features from medical images.

More complex architectures (like DenseNet) are used in real-world medical imaging applications. But for a small dataset and classroom setting, our custom architecture is ideal for learning.

To make this process modular and reusable, we’ll write a class called `ChestXRayNet` using PyTorch.

```python
import torch
from torch import nn

class ChestXRayNet(nn.Module):
    def __init__(self, dropout_rate=0.6):
        super(ChestXRayNet, self).__init__()
        
        # First convolutional block: 8 filters (3x3), followed by max pooling
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # Third and fourth blocks: 12 filters
        self.conv3 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        
        # Fifth and sixth blocks: 20 filters, 5x5 kernel
        self.conv5 = nn.Conv2d(12, 20, kernel_size=5, padding=2)
        self.pool5 = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        self.pool6 = nn.MaxPool2d(2)
        
        # Final convolutional layer with 50 filters
        self.conv7 = nn.Conv2d(20, 50, kernel_size=5, padding=2)
        
        # Global average pooling reduces each feature map to a single value
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Dense (fully connected) layers for classification
        self.fc1 = nn.Linear(50, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Conv blocks
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.pool5(self.relu(self.conv5(x)))
        x = self.pool6(self.relu(self.conv6(x)))
        
        # Final conv and GAP
        x = self.relu(self.conv7(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x

# Create the model
model = ChestXRayNet(dropout_rate=0.6)
```

:::::::::::::::::::::::::::::::::::::::  challenge

## Exercise

A) What is the purpose of using multiple convolutional layers in a neural network?  
B) What would happen if you skipped the pooling layers entirely?  

:::::::::::::::  solution

## Solution

A) Stacking convolutional layers allows the network to learn increasingly abstract features — early layers detect edges and textures, while later layers detect shapes or patterns.

B) Skipping pooling layers means the model retains high-resolution spatial information, but it increases computational cost and can lead to overfitting.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

Now let's build the model and view its architecture:

```python
import torch

# Set the seed for reproducibility
torch.manual_seed(42)

# Create the model
model = ChestXRayNet(dropout_rate=0.6)

# View the model architecture
print(model)
```

```output
ChestXRayNet(
  (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(8, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv4): Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv5): Conv2d(12, 20, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (pool5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv6): Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (pool6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv7): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (gap): AdaptiveAvgPool2d(output_size=1)
  (fc1): Linear(in_features=50, out_features=128, bias=True)
  (dropout): Dropout(p=0.6, inplace=False)
  (fc2): Linear(in_features=128, out_features=32, bias=True)
  (fc3): Linear(in_features=32, out_features=1, bias=True)
  (relu): ReLU()
  (sigmoid): Sigmoid()
)
```

:::::::::::::::::::::::::::::::::::::::  challenge

## Exercise

Increase the number of filters in the first convolutional layer from 8 to 16.  

- How does this affect the number of parameters in the model?
- What effect do you expect this change to have on the model’s learning capacity?

:::::::::::::::  solution

## Solution

In the `ChestXRayNet` class, locate this line in `__init__`:

```python
self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
```

Change it to:

```python
self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
```

This increases the number of filters (feature detectors), and therefore increases the number of learnable parameters. The model may be able to capture more features, improving learning, but it also risks overfitting and will take longer to train.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::::: keypoints

- Neural networks are composed of layers of neurons that transform inputs into outputs through learnable parameters.
- Activation functions introduce non-linearity and help neural networks learn complex patterns.
- Dense (fully connected) layers connect every neuron from one layer to the next and are commonly used in classification tasks.
- Convolutional layers apply filters to extract spatial features from images and are the core of convolutional neural networks (CNNs).
- Dropout helps reduce overfitting by randomly disabling neurons during training.

::::::::::::::::::::::::::::::::::::::::::::::::::


