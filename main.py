# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 1
#
# Using the MNIST data sets to train and test perceptrons.

import numpy
import pandas

EPOCH_MAX  = 50
TRAIN_FILE = "mnist_train.csv"
TEST_FILE  = "mnist_test.csv"

# Import training data

# Preprocess data 