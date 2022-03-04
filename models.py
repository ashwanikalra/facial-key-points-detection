## TODO: define the convolutional neural network architecture

import torch.nn as nn
import torch.nn.functional as F


# can use the below import should you choose to initialize the weights of your Net


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # TODO: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the key points
        # it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        # after conv layer 1, the output using formula with 0 padding (W-F`+2P)/S+1 = (224-5)/1+1 = 220
        # after max pooling, to above output, the output size is 220/2 = 110
        # define second conv layer, input image size is 110x110
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        # after conv layer 2, the output using formula with 0 padding (W-F+2P)/S+1 = (110-3)/1+1 = 108
        # after max pooling, to above output, the output size is 108/2 = 54
        # define third conv layer, input image size is 54x54
        self.conv3 = nn.Conv2d(64, 128, (2, 2))
        # after conv layer 3, the output using formula with 0 padding (W-F+2P)/S+1 = (54-2)/1+1 = 52
        # after max pooling, to above output, the output size is 52/2 = 26

        # Note that among the layers to add, consider including:
        # max pooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch
        # normalization) to avoid over fitting
        # define max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # define dropout layer
        self.drop = nn.Dropout(p=0.2)
        # define fully connected layer, input size is 26*26*128=5408
        self.fc1 = nn.Linear(128 * 26 * 26, 1024)
        self.fc2 = nn.Linear(1024, 136)

    def forward(self, x):
        # TODO: Define the feedforward behavior of this model
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu((self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
