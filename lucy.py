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

def get_data(inputs_file_path):
    loaded = np.load(inputs_file_path)
    inputs = loaded['a']
    norm_inputs = inputs/255.0
    final_inputs = np.float32(norm_inputs)
    return final_inputs


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
        self.input_size = 784 
        self.num_classes = 2 
        self.batch_size = 100 
        self.learning_rate = 0.5 

        self.W = np.zeros((self.input_size, self.num_classes))
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
        accumulated_loss = 0
        for j in range(self.batch_size):
            correct_answer = labels[j]
            given_probability = probabilities[j][correct_answer]
            accumulated_loss += -(np.log(given_probability))
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
        delta_W = np.zeros((self.input_size, self.num_classes))
        delta_B = np.zeros((1,self.num_classes))
        
        for image in range(self.batch_size):
            one_hot = np.zeros((1,self.num_classes))
            correct_image_answer = labels[image]
            one_hot[0][correct_image_answer] = 1 
            trans_inputs = np.transpose(np.reshape(inputs[image], (1, self.input_size)))
            delta_W +=  np.matmul(trans_inputs, (probabilities[image] - one_hot))
            delta_B += (probabilities[image] - one_hot)

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
        predictions = np.argmax(probabilities, axis=1)
        return np.mean(predictions == labels)

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
    num_iterations = int(len(train_inputs)/model.batch_size)
    for i in range(num_iterations):
        new_train_inputs = train_inputs[i * current_batch : (i * current_batch) + current_batch]
        new_train_labels = train_labels[i * current_batch : (i * current_batch) + current_batch]
        train_call_probabilities = model.call(new_train_inputs)
        train_w_grad, train_b_grad = model.back_propagation(new_train_inputs, train_call_probabilities, new_train_labels)
        model.gradient_descent(train_w_grad, train_b_grad)

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    model_test_probability = model.call(test_inputs)
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

    airplane_inputs = get_data('airplane.npz')
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
    model = Model()
    train(model, final_train_inputs, final_train_labels)
    accuracy = test(model, final_test_inputs, final_test_labels)
    print(accuracy)

if __name__ == '__main__':
    main()
