# Neural Network to Detect Handwritten Digits
Neural Network to Detect Handwritten Digits using python

## Installation and Setup

### Overview
This project based on book by [Michael Nielson](http://neuralnetworksanddeeplearning.com)

Various folders in this project are:

1. data:  It contains the mnist database
2. digits:  It has two folders *testing* and *myimage*
   * *testing*: It contains various images from 0-9 that can be used for the purpose of testing the neural network
   * *myimage*: It contains 9 images written by me.
3. src: src folder has 3 files:
    * *mnist_loader.py*: Used to load mnist data that is meant for training testing and validation. Mnist Database contains 60,000, 50,000 of them is used for training purpose and rest 10,000 is used for validation and testing
    * *network1.py*: Naive neural network uses concept of **cost function**, **backpropagation** and **stochastic gradient descent**
    * *network2.py*: Advancement of network1.py has additional feature **overfitting** and **crossEntropyCost function**, in addtion to features in networ1.py
  
4. models: It contains various models made by me, model5.txt is most efficient model of them.

## Dependencies
Following python libraries are required:

1. numpy: install using *pip install numpy*
2. PIL: install using *pip install pillow*

## Running
Running is based on the steps:

1. Load mnist data
2. Create the model file
3. Create an image file containing a handwritten number
4. Format the handwritten number to suit mnist format
5. Predict the integer 

#### 1. Create the model file
The easiest way is to cd to src directory where the python files are located. Then run:

1. python
2. **>>>** import mnist_loader  #This step is used to import mnist_loader.py
3. **>>>** training, validation, testing = mnist_loader.load_data_wrapper() #This step is used to load training validation and testing data
4. import network2 #This step is used to import network2.py
5. **>>>** net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost) # This step creates a neural network with 784 neuron in input layer, 30 neuron in 1st hidden layer, 10 neuron in output layer
6. **>>>** net.SGD(training_data, 30, 10, 0.5,<br>
  ... lmbda = 5.0,<br>
  ... filename=("../models/modelx.txt")<br>
  ... evaluation_data=validation_data,<br>
  ... monitor_evaluation_accuracy=True,<br>
  ... monitor_evaluation_cost=True,<br>
  ... monitor_training_accuracy=True,<br>
  ... monitor_training_cost=True)<br>

These steps will create and save neural network model in "../models/modelx.txt".

#### 2. Process any hand written image

In order to test any handwritten image we need to first process it so that it matches with mnist data, to process the image we follow these steps 

 1. convert 81.png -monochrome a1.png
 2. convert -resize 28x28 a1.png a1.png

#### 3. Loading the image

To load and test the image we follow following steps:

1. cd to the src directory
2. python
3. **>>>** import mnist_loader
4. **>>>** test_against = mnist_loader.imageread("../filepath/filename_of_image")

#### 4. Predict the Interger

To predict the interger written in image

1. **>>>** import network2
2. **>>>** network2.load(test_against, "../modelpath/modelname.txt") #modelpath is where the model was saved
