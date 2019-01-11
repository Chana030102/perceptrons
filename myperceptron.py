# myperceptron.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 1
# 
# Perceptron will be initialized with a specified number of inputs.
# Bias should not be included in input count argument. Bias will always be provided.

import numpy 
WEIGHT_LOW = -0.05
WEIGHT_HIGH = 0.05

class Perceptron:
    # Initialize with 
    def __init__(self, num_inputs, learn_rate):
        self.size = num_inputs
        self.learn_rate = learn_rate
        self.weights = numpy.random.uniform(low=WEIGHT_LOW,high=WEIGHT_HIGH,size=(num_inputs,))
        self.bias_weight = numpy.random.uniform(low=WEIGHT_LOW,high=WEIGHT_HIGH)

    # Evaluates inputs by summing the products of inputs and weights
    # Return -1 if size of inputs doesn't match initialized input size for Perceptron
    # Returns 1 if evaluates greater than 0
    # Returns 0 otherwise
    def evaluate(self,inputs):
        if len(inputs) != self.size:
            return -1
        
        val = numpy.sum(numpy.multiply(self.weights,inputs)) + self.bias_weight
        
        if val > 0:
            return 1
        else:
            return 0
    
    # Update weights for inputs if output didn't match target
    # Change of weight = (learn_rate)*(target-output)*(input)
    def updateWeights(self,target,output,inputs):
        delta = self.learn_rate*(target-output)
        delta_weights = delta*inputs
        self.weights = numpy.add(self.weights,delta_weights)
        self.bias_weight += delta
