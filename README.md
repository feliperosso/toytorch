# toytorch

This repository was initiated to achieve the following two goals:

1. Learn the detailed structure of the standard neural network architectures, their training and basic capabilities.
2. Learn how to use the Pytorch library for deep learning.

The result of this undertaking is the "toytorch" package, which implements the most common neural network tools from scratch, using a similar structure as the Pytorch library. Instead of using numpy to perform tensor manipulations, toytorch is built from torch tensors due to the following two advantages:

- Automatic differentiation via autograd.
- Cuda support to run computations on GPU.

In order to show what is possible with toytorch, we include three demos, in the form of Jupyter notebooks:

1. In [demo_1](https://github.com/feliperosso/toytorch/blob/main/demo_1/ConvolutionalCIFAR10.ipynb) we train a convolutional model to clasify the CIFAR10 image dataset.
2. In [demo_2](https://github.com/feliperosso/toytorch/blob/main/demo_2/2.%20LSTMLanguageModel%20(GitHub).ipynb) we train a LSTM language model to memorize and reproduce the lyrics of Bohemian Rhapsody.
3. In [demo_3](https://github.com/feliperosso/toytorch/blob/main/demo_3/Beatles_Transformer_model.ipynb) we train a Transformer language model on The Beatles lyrics and use it to generate new lyrics in their style.

Let us now briefly review the content of the main modules in the toytorch package.

### [func.py](https://github.com/feliperosso/toytorch/blob/main/toytorch/func.py)

This module defines the following basic functions that are useful for building neural networks:

- Cosine Learning Schedule
- Cross Entropy
- Log Softmax
- One Hot Encoder
- ReLU
- Sigmoid
- Softmax

### [layers.py](https://github.com/feliperosso/toytorch/blob/main/toytorch/layers.py)

This is the main module of the package, where the basic layers are constructed from scratch using torch tensors. Apart from basic tensor manipulations, the only torch capabilities that are used are: cuda, nn.Module and nn.Parameter. The layers defined in this module are:

- Convolution
- Dropout
- Embedding
- Layer Norm
- Linear
- LSTM
- Multihead Attention
- Pooling
- Positional Encoding
- TransformerDecoderLayer
- TransformerEncoderLayer

### [models.py](https://github.com/feliperosso/toytorch/blob/main/toytorch/models.py)

This module combines the layers defined in layers.py into fully fledged neural networks. The torch capabilities used in this module are: cuda, nn.Module, nn.ModuleList, autograd and optimizer. The end result are the following three models:

- A convolutional network model that can be trained to classify images.
- A language model based based on LSTM layers, which can be used to train on any text and then generate text in the same style.
- A language model based based on the Transformer, which can be used to train on any text and then generate text in the same style.

To illustrate their functionalities, we have included explicit implementations of each in the demos.

### [data_utils.py](https://github.com/feliperosso/toytorch/blob/main/toytorch/data_utils.py)

This module contains two tools for text processing, one based on a simple word tokenizer and another on a Byte Pair Encoding (BPE). Each of them comes with an encoder/decoder as well as an iterable structure that can be used directly into Pytorch's DataLoader in order to feed the model with the data during training.

### [datasets](https://github.com/feliperosso/toytorch/tree/main/toytorch/datasets)

Finally, we have included some datasets that can be used for training and testing the models.
