# main.py
# 
# Aaron Chan
# CS 445 Machine Learning (Winter 2019)
# Homework 1
#
# Using the MNIST data sets to train and test perceptrons.
# Assumes first column is target info and other columns are input data
#
# This assignment requires the following:
# - accuracy based off training and test data sets
# - accuracy before training and after each epoch
# - perceptron with greatest output is treated to be the prediction for network
# - confusion matrix for test data set after training is completed
# - perform the above with three different learning rates: 0.01, 0.1, 1

import numpy
import pandas
import myperceptron

EPOCH_MAX  = 2
INPUT_MAX  = 255
learning_rate = [0.01, 0.1, 1]

TRAIN_FILE = "../mnist_train.csv"
TEST_FILE  = "../mnist_test.csv"
output = [None]*10 # used to store outputs and evaluate the prediction of network
accuracy = pandas.DataFrame(0,index=range(0,EPOCH_MAX+1),columns=['test_c','test_i','train_c','train_i'])

# set up confusion matrix: rows=actual, col=predicted
confu  = pandas.DataFrame(0,index=range(0,10),columns=range(0,10)) 

# Import data
train_data = pandas.read_csv(TRAIN_FILE,header=None)
test_data  = pandas.read_csv(TEST_FILE ,header=None)

# Preprocess data 
train_data.sample(frac=1)      # shuffle training data
train_target = train_data[0] # Save targets as a separate dataframe/array
train_data.drop(columns=0)     # Remove column with target info
train_data /= INPUT_MAX        # scale inputs between 0 and 1 by dividing by input max value

test_target = test_data[0] # Save targets as a separate dataframe/array
test_data.drop(columns=0)    # Remove column with target info
test_data /= INPUT_MAX       # scale inputs between 0 and 1 by dividing by input max value

input_size = len(train_data.columns) # how many inputs are there

for rate in learning_rate:
    # new network of perceptrons
    net = [myperceptron.Perceptron(input_size) for i in range(0,10)]

    # Epoch 0 accuracy test for train data
    for i in range(0,len(train_data)):
        for j in range(0,10):
            output[j] = net[j].evaluate(train_data.loc[i])
        
        if(train_target[i] == output.index(max(output))):
            accuracy['train_c'][0]+=1
        else:
            accuracy['train_i'][0]+=1
        
    # Epoch 0 accuracy test for test data
    for i in range(0,len(test_data)):
        for j in range(0,10):
            output[j] = net[j].evaluate(test_data.loc[i])

        if(output.index(max(output)) == test_target[i]):
            accuracy['test_c'][0]+=1
        else:
            accuracy['test_i'][0]+=1

    # Start training and record accuracy after each epoch
    for e in range(0,EPOCH_MAX+1):

        # Loop through each row of training data
        for x in range(0,len(train_data)):
            # Loop through each perceptron
            for y in range(0,10):
                output[y] = net[y].evaluate(train_data.loc[x])

            # Evaluate outputs. Greatest value is prediction and equals 1. Others are 0
            prediction = output.index(max(output))
            for i in range(0,10):
                if(i==prediction):
                    output[i] = 1
                else:
                    output[i] = 0
            
            # Update weights based off prediction
            for y in range(0,10):
                if(y==train_target[x]):
                    net[y].updateWeights(1,output[y],train_data.loc[x],rate)
                else:
                    net[y].updateWeights(0,output[y],train_data.loc[x],rate)

            # Evaluate and test accuracy at each epoch with training data
            for i in range(0,len(train_data)):
                for j in range(0,10):
                    output[j] = net[j].evaluate(train_data.loc[i])
                
                if(output.index(max(output)) == train_target[i]):
                    accuracy['train_c'][e]+=1
                else:
                    accuracy['train_i'][e]+=1
                
            # Evaluate and test accuracy at each epoch with test data
            for i in range(0,len(test_data)):
                for j in range(0,10):
                    output[j] = net[j].evaluate(test_data.loc[i])
                
                if(output.index(max(output)) == test_target[i]):
                    accuracy['test_c'][e]+=1
                else:
                    accuracy['test_i'][e]+=1
    
    # Finished training for this learning rate. Construct confusion matrix
    for i in range(0,len(test_data.columns)):
        for j in range(0,10):
            output[j] = net[j].evaluate(test_data.loc[i])
        
        # Confusion matrix --> confu[actual][prediction]
        confu[test_target[i]][output.index(max(output))]+=1
    
    # Output accuracy and confusion matrix to CSV file
    accuracy.to_csv('accuracy, rate='+str(rate)+'.csv')
    confu.to_csv('confusionmatrix, rate='+str(rate)+'.csv')

    # Reset matrix and accuracy table values to zero for next learning rate
    accuracy[:] = 0
    confu[:] = 0