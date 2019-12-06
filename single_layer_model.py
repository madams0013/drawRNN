import numpy as np
import math
from matplotlib import pyplot as plt


def get_data(inputs_file_path):
    """
    Opens up compressed file where the data is stored
    :param inputs: file paths for the different sets of images
    :return: inputs that are going to be used for the model
    """  
    loaded = np.load(inputs_file_path)
    inputs = loaded['a']
    norm_inputs = inputs/255.0
    final_inputs = np.float32(norm_inputs)
    return final_inputs

def preprocess(airplane_file, ant_file, cake_file):
    """
    Pre-processes the data used by the model.
    :param inputs: file paths for the different sets of images
    :return: train inputs, train labels, test inputs and test labels used by the 
    model 
    """    
    airplane_inputs = get_data(airplane_file)
    ant_inputs = get_data(ant_file)
    cake_inputs = get_data(cake_file)

    airplane_inputs = airplane_inputs[:50000]
    ant_inputs = ant_inputs[:50000]
    cake_inputs = cake_inputs[:50000]
    
    airplane_break_length = math.floor(0.8*len(airplane_inputs))
    ant_break_length = math.floor(0.8*len(ant_inputs))
    cake_break_length = math.floor(0.8*len(cake_inputs))

    airplane_train_inputs = airplane_inputs[:airplane_break_length]
    ant_train_inputs = ant_inputs[:ant_break_length]
    cake_train_inputs = cake_inputs[:cake_break_length]

    airplane_test_inputs = airplane_inputs[airplane_break_length:]
    ant_test_inputs = ant_inputs[ant_break_length:]
    cake_test_inputs = cake_inputs[cake_break_length:]

    airplane_train_labels = [0 for _ in range(len(airplane_train_inputs))]
    ant_train_labels = [1 for _ in range(len(ant_train_inputs))]
    cake_train_labels = [2 for _ in range(len(cake_train_inputs))]

    airplane_test_labels = [0 for _ in range(len(airplane_test_inputs))]
    ant_test_labels = [1 for _ in range(len(ant_test_inputs))]
    cake_test_labels = [2 for _ in range(len(cake_test_inputs))]

    final_train_inputs = np.concatenate((airplane_train_inputs, ant_train_inputs, cake_train_inputs))
    final_test_inputs = np.concatenate((airplane_test_inputs, ant_test_inputs, cake_test_inputs))
    final_train_labels = np.concatenate((airplane_train_labels, ant_train_labels, cake_train_labels))
    final_test_labels = np.concatenate((airplane_test_labels, ant_test_labels, cake_test_labels))

    train_indices = np.arange(len(final_train_inputs))
    np.random.shuffle(train_indices)
    final_train_inputs = final_train_inputs[train_indices]
    final_train_labels = final_train_labels[train_indices]

    test_indices = np.arange(len(final_test_inputs))
    np.random.shuffle(test_indices)
    final_test_inputs = final_test_inputs[test_indices]
    final_test_labels = final_test_labels[test_indices]
    return final_train_inputs, final_train_labels, final_test_inputs, final_test_labels
    


class Model:
    """
    Model class contains the functionality for the single-layer
    Neural Network for the QuickDraw! Dataset
    """

    def __init__(self):
        self.input_size = 784
        self.num_classes = 3
        self.batch_size = 100
        self.learning_rate = 0.5

        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros((1,self.num_classes))

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized batch of images
        :return: probabilities for each class for each image
        """
        logits = np.matmul(inputs, self.W) + self.b
        call_probabilities = np.exp(logits)/np.sum(np.exp(logits), axis = 1, keepdims=True)
        return call_probabilities

    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param probabilities matrix matrix 
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
        after one forward pass and loss calculation. 
        :param inputs: batch inputs
        :param probabilities: probabilities matrix
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
        Calculates the model's accuracy 
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
    Tests the model on the test inputs and labels. 
    :param test_inputs: QuickDraw! test data (all images to be tested)
    :param test_labels: QuickDraw! test labels (all corresponding labels)
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
    Read in Quick!Draw data, initializes model, and does training and testing. 
    :return: None
    '''

    train_inputs, train_labels, test_inputs, test_labels = preprocess('dataset/airplane.npz', 'dataset/ant.npz', 'dataset/birthday_cake.npz')

    model = Model()
    train(model, train_inputs, train_labels)
    accuracy = test(model, test_inputs, test_labels)
    visualize_results(test_inputs[10:20], model.call(test_inputs[10:20]), test_labels[10:20])
    print("Accuracy of our model", accuracy)

if __name__ == '__main__':
    main()
