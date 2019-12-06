import numpy as np
import math
import logging
import os
import cloudstorage as gcs
import gzip
from matplotlib import pyplot as plt
import tensorflow as tf
import torch as torch
import torch.nn as nn


def get_data(inputs_file_path):
    loaded = np.load(inputs_file_path)
    inputs = loaded['a']
    norm_inputs = inputs/255.0
    final_inputs = np.float32(norm_inputs)
    return final_inputs

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args 
    def forward(self, x):
        return x.view(self.shape)

class Model(nn.Module):
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
        super(Model, self).__init__()
        self.num_classes = 2 
        self.batch_size = 100 
        self.learning_rate = 0.01
        self.cross_loss = nn.CrossEntropyLoss()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            Reshape(-1, 4*4*50),
            nn.Linear(4*4*50,500),
            nn.ReLU(),
            nn.Linear(500, 10),
            nn.ReLU(),
            nn.Linear(10,self.num_classes),
            nn.Softmax(dim=1)
            )
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        inputs = np.reshape(inputs, (len(inputs),1, 28, 28))
        inputs = torch.tensor(inputs)
        return self.model(inputs)

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
        return self.cross_loss(probabilities, labels)/self.batch_size

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
    loss = nn.CrossEntropyLoss()

    for i in range(num_iterations):
        new_train_inputs = train_inputs[i * current_batch : (i * current_batch) + current_batch]
        new_train_labels = train_labels[i * current_batch : (i * current_batch) + current_batch]
        train_call_probabilities = model.call(new_train_inputs)
        labels_tensor = torch.tensor(new_train_labels)
        loss = model.loss(train_call_probabilities, labels_tensor)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
    return None

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    model_test_probability = model.call(test_inputs).detach().numpy()
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

    airplane_inputs = get_data('dataset/airplane.npz')
    ant_inputs = get_data('dataset/ant.npz')
    
    
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

    train_indices = np.arange(len(final_train_inputs))
    np.random.shuffle(train_indices)
    final_train_inputs = final_train_inputs[train_indices]
    final_train_labels = final_train_labels[train_indices]

    test_indices = np.arange(len(final_test_inputs))
    np.random.shuffle(test_indices)
    final_test_inputs = final_test_inputs[test_indices]
    final_test_labels = final_test_labels[test_indices]
    
    model = Model()
    print("before training")
    train(model, final_train_inputs, final_train_labels)
    print("im after training")
    print("starting testing")
    accuracy = test(model, final_test_inputs, final_test_labels)
    visualize_results(np.array(final_test_inputs[10:20]), np.array(model.call(final_test_inputs[10:20]).detach().numpy()), np.array(final_test_labels[10:20]))
    print(accuracy)

if __name__ == '__main__':
    main()
