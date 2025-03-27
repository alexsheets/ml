import numbers
import random
import math

#---------------------------------------------------

# GOAL OF BACKPROPAGATION: optimize weights so neural network can learn how to correctly map arbitrary inputs to outputs
# PROCEDURE OF BACKPROPAGATION:
# --> forward pass: input data initially fed through; each neuron performs calculations based on the weights and biases of these nodes, ultimately producing output
# --> error calculation: predicted output compared to the actual output and error/loss is calculated
# --> backward pass (backpropagation): error propagated backward through network layers starting at the output layer
# --> gradient calculation: backward pass uses chain rule of calculus to calculate the derivative of the loss function with respect to each bias and weight in the network
# --> weight/bias adjustment: gradients indicate direction and magnitude of change needed for each weight and bias to minimize error
# --> iterative process: entire process repeated for multiple epochs until the performance stabilizes/error minimized

# creating a neural network with two inputs, two hidden neurons, two output neurons (neurons all include bias)
# hidden neuron: node within a hidden layer of a neural network which is neither input nor output layer
# output neuron: node in final layer of neural network which produces model's prediction or result

# initial weights, biases, training inputs and outputs: https://mattmazur.com/wp-content/uploads/2018/03/neural_network-9.png

# Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

class BackPropagation:
    LEARNING_RATE = 0.5


class NeuronLayer:
    def __init__(self, num, bias):
        self.bias = bias
        self.neurons = []
        for i in range(num):
            self.neurons.append(Neuron(self.bias))


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    # forward pass: 
    # figure out total net input to each hidden layer neuron
    # squash the total net input using activation (logistic here) function
    # repeat process with output layer neurons

    def net_input(self):
        # simply calculate total input to this neuron and return
        # include bias in return value
        net = 0
        for i in self.inputs:
            net += self.inputs[i] * self.weights[i]
        return net + self.bias
    
    # squash function to use on output of neuron
    def squash(self, net_input):
        # formula of form 1 / 1 + e^-net
        return 1 / (1 + math.exp(-net_input))
    
    def calc_output(self, inputs):
        self.inputs = inputs
        # repeat the process of calculating net input and squashing for each
        # using the output from hidden layer neurons as inputs
        self.output = self.squash(self.net_input())
        return self.output
        
    # calc total error for each output neuron using squared error func
    # total err = 1/2 * (target - output)^2
    def calc_error(self, target_output):
        return (1/2) * (target_output - self.output) ** 2
    
    # need partial derivative/gradient descent with respect to total net input, output
    # our goal in backpropagatoin: update weights in network so they cause actual
    # output to be closer to target output

    # we have partial derivative of error with respect to output and derivative of output with respect to total net input
    # we can calculate the partial derivative of error with respect to total net input
    def calc_pd_err_total_net_input(self, target_output):
        return self.calc_pd_err_total_net_input(target_output) * self.calc_pd_total_net_input()
    
    # total net input into neuron squashed using logistic function to calculate neurons output
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # j represents the output of the neurons in whatever layer we're looking at, i represents the layer below it
    def calc_pd_total_net_input(self):
        return self.output * (1 - self.output)