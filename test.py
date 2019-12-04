import numpy as np
import math
import logging
import os
import cloudstorage as gcs
import webapp2
import gzip

from google.appengine.api import app_identity

def read_file(self, filename):
  self.response.write('Reading the full file contents:\n')

  gcs_file = gcs.open(filename)
  contents = gcs_file.read()
  gcs_file.close()
  self.response.write(contents)

def preprocess(inputs_file_path):
    inputs = np.load(inputs_file_path)
    inputs_array = np.array((len(inputs), 784), dtype=np.float32)
    inputs_array = np.array(inputs, dtype=np.float32)
    inputs_array /= 255.0
    return inputs_array

def get_data(inputs_file_path, labels_file_path, num_examples):
	"""
	Takes in an inputs file path and labels file path, unzips both files, 
	normalizes the inputs, and returns (NumPy array of inputs, NumPy 
	array of labels). Read the data of the file into a buffer and use 
	np.frombuffer to turn the data into a NumPy array. Keep in mind that 
	each file has a header of a certain size. This method should be called
	within the main function of the model.py file to get BOTH the train and
	test data. If you change this method and/or write up separate methods for 
	both train and test data, we will deduct points.
	:param inputs_file_path: file path for inputs, something like 
	'MNIST_data/t10k-images-idx3-ubyte.gz'
	:param labels_file_path: file path for labels, something like 
	'MNIST_data/t10k-labels-idx1-ubyte.gz'
	:param num_examples: used to read from the bytestream into a buffer. Rather 
	than hardcoding a number to read from the bytestream, keep in mind that each image
	(example) is 28 * 28, with a header of a certain number.
	:return: NumPy array of inputs as float32 and labels as int8
	"""
	inputs = None
	labels = None
	
	with open(inputs_file_path, 'rb') as f1, gzip.GzipFile(fileobj=f1) as bytestream1:
		bytestream1.read(16)
		inputs_buffer = bytestream1.read(784 * num_examples)
		inputs = np.frombuffer(inputs_buffer, dtype=np.uint8)
	
	with open(labels_file_path, 'rb') as f2, gzip.GzipFile(fileobj=f2) as bytestream2:
		bytestream2.read(8)
		labels_buffer = bytestream2.read(num_examples)
		labels = np.frombuffer(labels_buffer, dtype=np.uint8)
	
	norm_inputs = inputs/255.0
	final_inputs = np.float32(norm_inputs)
	final_labels = np.int8(labels)
	final_inputs = np.reshape(final_inputs, (num_examples, 784))
	return final_inputs, final_labels

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        self.batch_size = 100 # setting batch size variable
        self.num_classes = 2 # setting number of possible labels to 10
        self.learning_rate = 0.5
        self.input_size = 784 #size of image vectors

        self.W = None
        self.b = None
        self.W = np.zeros((self.input_size, self.num_classes), dtype=np.float32) # initializing weights array
        self.b = np.zeros(self.num_classes, dtype=np.float32) # initializing bias array

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        probabilities = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
        for i in range(self.batch_size):
            unbiased_logits = np.matmul(inputs[i], self.W)
            logits = np.add(unbiased_logits, self.b)
            exp_sum = np.sum(np.exp(logits))
            exp_logits = np.exp(logits)
            for l in range(len(logits)):
                probabilities[i][l] = exp_logits[l] / exp_sum
        return probabilities

    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be decreasing with every training loop (step).
        NOTE: This function is not actually used for gradient descent
        in this assignment, but is a sanity check to make sure model
        is learning.
        :param probabilities: matrix that contains the probabilities
        per class per image of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        loss = np.zeros(self.batch_size, dtype=np.float32) # initializing loss array
        for i in range(self.batch_size):
            loss[i] = (-1)*np.log(probabilities[i][labels[i]])
        average = np.mean(loss)
        return average

    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass and loss calculation. The learning
        algorithm for updating weights and biases mentioned in
        class works for one image, but because we are looking at
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # this function is called every time we go through a batch
        delta_W = np.zeros((self.input_size, self.num_classes))
        delta_B = np.zeros(self.num_classes)
        for i in range(self.batch_size):
            y_j = np.zeros(self.num_classes, dtype=int)
            y_j[labels[i]] = 1
            transposed_input = np.reshape(np.transpose(inputs[i]), (self.input_size,1))
            delta_probabilities = np.reshape(probabilities[i]-y_j, (1,self.num_classes))
            delta_W += np.matmul(transposed_input, delta_probabilities)
            delta_B += (probabilities[i]-y_j)
        delta_W /= self.batch_size
        delta_B /= self.batch_size
        return delta_W, delta_B




    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        model_labels = np.zeros(self.batch_size, dtype=np.float32)
        for i in range(self.batch_size):
            if labels[i] == np.argmax(probabilities[i]):
                model_labels[i] = 1
        return np.mean(model_labels)



    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        self.W -= self.learning_rate*gradW
        self.b -= self.learning_rate*gradB

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    start_point = 0
    end_point = model.batch_size
    while end_point <= len(train_inputs):
        inputs_batch = train_inputs[start_point:end_point]
        labels_batch = train_labels[start_point:end_point]
        probabilities = model.call(inputs_batch)
        weights, bias = model.back_propagation(inputs_batch, probabilities, labels_batch)
        model.gradient_descent(weights, bias)
        start_point = end_point
        end_point = end_point + model.batch_size

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    probabilities = model.call(test_inputs)
    return model.accuracy(probabilities, test_labels)

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in MNIST data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    '''

    # load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    airplane_inputs = preprocess('full_numpy_bitmap_airplane.npy')
    ant_inputs = preprocess('full_numpy_bitmap_ant.npy')

    airplane_break_length = math.floor(0.8*len(airplane_inputs))
    ant_break_length = math.floor(0.8*len(ant_inputs))

    airplane_train_inputs = airplane_inputs[:airplane_break_length]
    ant_train_inputs = ant_inputs[:ant_break_length]

    airplane_test_inputs = airplane_inputs[airplane_break_length:]
    ant_test_inputs = ant_inputs[ant_break_length:]

    airplane_train_labels = [0 for _ in range(len(airplane_train_inputs))]
    ant_train_labels = [1 for _ in range(len(ant_train_inputs))]

    airplane_test_labels = [0 for _ in range(len(airplane_test_inputs))]
    ant_test_labels = [1 for _ in range(len(ant_test_inputs))]

    final_train_inputs = np.concatenate((airplane_train_inputs, ant_train_inputs))
    final_test_inputs = np.concatenate((airplane_test_inputs, ant_test_inputs))
    final_train_labels = np.concatenate((airplane_train_labels, ant_train_labels))
    final_test_labels = np.concatenate((airplane_test_labels, ant_test_labels))
    print(final_train_inputs.shape)
    # # Create Model
    model = Model()
    # # Train model by calling train() ONCE on all data
    train(model, final_train_inputs, final_train_labels)
    # # Test the accuracy by calling test() after running train()
    accuracy = test(model, final_test_inputs, final_test_labels)
    # print(accuracy)
    print(accuracy)

if __name__ == '__main__':
    main()
