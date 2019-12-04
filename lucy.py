import numpy as np
import math
import logging
import os
import cloudstorage as gcs
import gzip

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
        # Initialize all hyperparametrs to those mentioned in the handout 
        self.input_size = 784 # Size of image vectors
        self.num_classes = 2 # Number of classes/possible labels
        self.batch_size = 100 # Number of images per each batch
        self.learning_rate = 0.5 #Learning rate mentioned in the handout

        # Initialize weights to be 784 x 10 
        self.W = np.zeros((self.input_size, self.num_classes))
        # Initialize the bias vector to be 1x10 
        self.b = np.zeros((1,self.num_classes))

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """

        logits = np.matmul(inputs, self.W) + self.b
        call_probabilities = np.exp(logits)/np.sum(np.exp(logits), axis = 1, keepdims=True)
        return call_probabilities

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
        # Create a counter to accumulate the cross entropy loss for my batch 
        accumulated_loss = 0
        #Go through every image in my bath size 
        for j in range(self.batch_size):
            # Get the "true" number that the model should output  
            correct_answer = labels[j]
            # Use the number to index into the probabilities array, to get the model's probability of 
            # outputing that number. 
            given_probability = probabilities[j][correct_answer]
            # Calculate the cross-entropy loss
            accumulated_loss += -(np.log(given_probability))
        # Average the accumulated loss by dividing by the size of the batch 
        return float(accumulated_loss)/self.batch_size

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
        # Create a "place-holder" matrix to store the changes in my weights and biases throughout my batch 
        delta_W = np.zeros((self.input_size, self.num_classes))
        delta_B = np.zeros((1,self.num_classes))
        
        #Go through all of the images in my batch
        for image in range(self.batch_size):
            
            # Create a one hot vector made up of all zeros. This one hot vector is of size 1 x 10
            one_hot = np.zeros((1,self.num_classes))
            
            # Because my batch size is 100, and my labels matrix has 100 entries, I can use the current 
            # index of my loop to get the "answer" of what the model should output 
            correct_image_answer = labels[image]
            
            # Since the "correct_image_answer" is an index between 0 and 9, we can index into our one hot 
            # vector and update the entry at that index to be 1. This is a vector representing what the
            # correct answer of the model should be. 
            one_hot[0][correct_image_answer] = 1 
            
            # Whenever I do inputs[image], I get a matrix with a shape of 784, . While the shape that I 
            # want for inputs[image] is 784 x 1, just having the shape be 784, was causing troubles. 
            # Therefore, I reshaped my matrix to 1 x 784 (what it should be originally) to make sure that
            # getting the transpose would result in a shape of 784 x 1 instead of 784, . 
            trans_inputs = np.transpose(np.reshape(inputs[image], (1, self.input_size))) #This gets an entire row, so 784 entries 
            
            # Update changes in W by following the formula seen in lecture 
            delta_W +=  np.matmul(trans_inputs, (probabilities[image] - one_hot))
            
            # Update the bias similar to the weights, except for the multiplication with the inputs. 
            delta_B += (probabilities[image] - one_hot)

        #Get the average of the weights and the biases
        average_W = delta_W/float(self.batch_size)
        average_B = delta_B/float(self.batch_size)
        
        return average_W, average_B


    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # Keep a counter of the number of times that my model correctly categorized an image 
        num_successes = 0
        # Go through the matrix of probabilities for my test data 
        for j in range(len(probabilities)):
            # Get the "correct answer" that the model should output 
            correct_number = labels[j]
            # Get the index of the largest probability in a given row of the probability matrix 
            predicted_number = np.argmax(probabilities[j])
            # Check if the output of my model matched the correct answer 
            if (predicted_number == correct_number):
                num_successes += 1
        # Calculate the batch accuracy by dividing by the batch size 
        final_accuracy = float(num_successes)/self.batch_size
        return final_accuracy



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
    current_batch = model.batch_size
    # Get the number of times that our model is going to run 
    num_iterations = int(len(train_inputs)/model.batch_size)
    # Iterate over the training inputs and labels, in model.batch_size increments
    for i in range(num_iterations):
        # Get the train inputs and labels by splicing the data 
        new_train_inputs = train_inputs[i * current_batch : (i * current_batch) + current_batch]
        new_train_labels = train_labels[i * current_batch : (i * current_batch) + current_batch]

        # For every batch, compute then descend the gradients for the model's weights
        train_call_probabilities = model.call(new_train_inputs)
        train_w_grad, train_b_grad = model.back_propagation(new_train_inputs, train_call_probabilities, new_train_labels)
        model.gradient_descent(train_w_grad, train_b_grad)
        # The model's loss was printed here. It went from 2 down to 0.35

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # Iterate over the testing inputs and labels
    model_test_probability = model.call(test_inputs)
    print(model_test_probability)
    # Return accuracy across testing set
    model_test_accuracy = model.accuracy(model_test_probability, test_labels)
    return model_test_accuracy

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
