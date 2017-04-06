"""
mnist_loader

The library to load the MNIST image data. For details of the data structures that are returned, see the doc sting for ``load_data`` and ``load_data_wrapper`` In practice, ``load_data_wrapper`` is the function usually called by our neural network code"""

import cPickle
import gzip
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    """ Returns MNIST data as a tuple containing the trainig data, validation data and the test data.

    The training_data is returned as a tuple with two entries. The first entry contains the actual training images. This is a numpy ndarray with 50,000 entries. Each entry is in turn a numpy ndarray with 784 values, representing 28*28 = 784 pixels in a single MNIST image.

    The second entry in the training_data tuple is the numpy ndarray containing 50,000 entries. Those entries are just the digit values (0..9) for the corresponding images contained in the first entry of the tuple.

    The validation_data and test_data are similar, except each contains only 10,000 images

    This is a nice data format but for use in neural networks it's helpful to modify the format of the training_data a little. That's done in the wrapper function load_data_wrapper
    """

    f = gzip.open('../data/mnist.pkl.gz','rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data,validation_data,test_data)

def load_data_wrapper():
    """ Return a tuple containg (training_data,validation_data,test_data) Based on load_data but the format is more convinient for use in out implementation of neural network.

    In particular training_data is a list of 50,000 2 tuples (x,y)
    x is a 784 dimensional numpy.ndarray containing the input image. y is a 10 dimensional numpy.ndarray representing the unit vector corresponding to the correct digit for x

    validation_data and test_data are lists containing 10,000 2 tuple (x,y). In each case x is a 784 dimensional numpy.ndarray containing the input image and y is the corresponding classification i.e the digit values corresponding to x

     Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d,va_d,te_d = load_data()
    #print (tr_d)
    training_inputs = [np.reshape(x,(784,1)).astype('float32')/255 for x in tr_d[0]]
    training_results = [vectorized_results(y) for y in tr_d[1]]
    training_data = zip(training_inputs,training_results)
    validation_inputs = [np.reshape(x,(784,1)).astype('float32')/255 for x in va_d[0]]
    validation_data = zip(validation_inputs,va_d[1])
    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = zip(test_inputs,te_d[1])
    return (training_data,validation_data,test_data)
def imageread(filepath):
    img=Image.open(filepath)
    #use following commands to convert some png image to required image format
    #convert 81.png -monochrome a1.png
    #convert -resize 28x28 a1.png a1.png
    #img.show()
    img=img.resize((28,28))
    iar=np.array(img)
    #print test_inputs
    #print iar
    test_against = np.reshape(iar,(784,1)).astype('float32')/255
    #print test_against
    #low_value_indices = test_against < 0.3
    #test_against[low_value_indices] = 0.0
    #high_value_indices = test_against >= 0.3
    #test_against[high_value_indices] = 1.0
    #img1 = Image.fromarray(test_against.reshape(28,28))
    #img1.show()
    #print test_against
    return test_against

def vectorized_results(j):
    """Return a 10 D vector with 1.0 in jth position and zeroes elsewhere, used to convert a digit (0..9) into corresponding desired output from the neural network"""

    e = np.zeros((10,1))
    e[j] = 1.0
    return e
