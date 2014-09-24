#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy
from utils import *


class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0

        # self.params = [self.W, self.b]

    def train(self, lr=0.1, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input

        # p_y_given_x = sigmoid(numpy.dot(self.x, self.W) + self.b)
        p_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x
        
        self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * numpy.mean(d_y, axis=0)
        
        # cost = self.negative_log_likelihood()
        # return cost

    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)
        
#         print len(sigmoid_activation), ' ', len(self.y)
#         print type(sigmoid_activation), ' ', type(self.y)
#         temp = numpy.log(sigmoid_activation) * self.y[:, None]
        
        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        
#         cross_entropy = - numpy.mean(
#             numpy.sum(numpy.log(sigmoid_activation) * self.y[:, None] +
#              numpy.log(1 - sigmoid_activation) * (1 - self.y[:, None]),
#                       axis=1))

        return cross_entropy


    def predict(self, x, y):
        
        print softmax(numpy.dot(x, self.W) + self.b)
        return numpy.mean(numpy.equal(numpy.argmax(softmax(numpy.dot(x, self.W) + self.b), axis=1), y))
        #return numpy.mean(numpy.equal(numpy.argmax(sigmoid(numpy.dot(x, self.W) + self.b), axis=1), y))
    
    
 
                


def test_lr2(learning_rate=0.1, n_epochs=50, dataset='mnist.pkl.gz'):
    # training data
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # cutting number of instances for training and testing
    n_train = 5000 # 50000
    n_test = 1000 # 10000
    train_set_x = numpy.array(train_set_x[0:n_train])
    train_set_y = numpy.array(train_set_y[0:n_train])
    test_set_x = numpy.array(test_set_x[0:n_test])
    test_set_y = numpy.array(test_set_y[0:n_test])
    
    train_set_y = numpy.array(ylabelToArray(train_set_y, 10))
    
    

    
    
    # print details
#     print train_set_x.shape, train_set_y_array.shape, valid_set_x.shape, valid_set_y.shape
#     print numpy.max(valid_set_x) 
#     print numpy.min(valid_set_x)
#     print numpy.mean(valid_set_x, axis=0)
#     print numpy.std(valid_set_x, axis=0)

    # construct LogisticRegression
    classifier = LogisticRegression(input=train_set_x, label=train_set_y, n_in=28*28, n_out=10)

    # train
    for epoch in xrange(n_epochs):
        classifier.train(input=train_set_x, lr=learning_rate)
        cost = classifier.negative_log_likelihood()
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        learning_rate *= 0.95

    print >> sys.stderr, classifier.predict(test_set_x, test_set_y)
    


def test_lr(learning_rate=0.01, n_epochs=200):
    # training data
    x = numpy.array([[1,1,1,0,0,0],
                     [1,0,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,1,1,1,0],
                     [0,0,1,1,0,0],
                     [0,0,1,1,1,0]])
    y = numpy.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]])


    # construct LogisticRegression
    classifier = LogisticRegression(input=x, label=y, n_in=6, n_out=2)

    # train
    for epoch in xrange(n_epochs):
        classifier.train(lr=learning_rate)
        cost = classifier.negative_log_likelihood()
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        learning_rate *= 0.95


    # test
    x = numpy.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0],
                     [1, 1, 1, 1, 1, 0]])

    print >> sys.stderr, classifier.predict(x)


if __name__ == "__main__":
    test_lr2()
