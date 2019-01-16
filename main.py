# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 1
#
# Using the MNIST data sets to train and test perceptrons.
# Assumes first column is target info and other columns are input data

import numpy
import pandas

EPOCH_MAX  = 50
INPUT_MAX  = 255
TRAIN_FILE = "mnist_train.csv"
TEST_FILE  = "mnist_test.csv"
output = [None]*10 # used to store outputs and evaluate the prediction of network

# set up confusion matrix: rows=actual, col=predicted
confu  = pandas.DataFrame(0,index=range(0,10),columns=range(0,10)) 

# Import data
train_data = pandas.read_csv(TRAIN_FILE,header=None)
test_data  = pandas.read_csv(TEST_FILE ,header=None)

# Preprocess data 
train_target = train_data[[0]] # Save targets as a separate dataframe/array
train_data.drop(columns=0)     # Remove column with target info
train_data /= INPUT_MAX        # scale inputs between 0 and 1 by dividing by input max value

test_target = test_data[[0]] # Save targets as a separate dataframe/array
test_data.drop(columns=0)    # Remove column with target info
test_data /= INPUT_MAX       # scale inputs between 0 and 1 by dividing by input max value

input_size = len(train_data.columns) # how many inputs are there

# Training
# evaluate
# update weights --> if(p == net[target[i])