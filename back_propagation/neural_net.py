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


class BackPropagation:
    LEARNING_RATE = 0.5