#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Deep Belief Nets (DBN)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007


   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials

'''

import time
import sys
import numpy
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from RBM import RBM
from utils import *


import PIL.Image


# import theano
# import theano.tensor as T



class DBN(object):

    
    def __init__(self, input=None, label=None,\
                 n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2,\
                 numpy_rng=None):
        
        self.x = input
        self.y = label

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        
        assert self.n_layers > 0


        # construct multi-layer
        for i in xrange(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()
                
            # construct sigmoid_layer
            sigmoid_layer = HiddenLayer(input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layer_sizes[i],
                                        numpy_rng=numpy_rng,
                                        activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)


            # construct rbm_layer
            rbm_layer = RBM(input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,     # W, b are shared
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)


        # layer for output using Logistic Regression
        self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()

#         print 'self.finetune_cost: ', self.finetune_cost



    def pretrain(self, lr=0.1, k=1, epochs=1000, batch_size=-1):


  
        
        pretaining_start_time = time.clock()
        # pre-train layer-wise
         
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i-1].sample_h_given_v(layer_input)
             
            rbm = self.rbm_layers[i]
#             print 'layer_input', layer_input
             
            for epoch in xrange(epochs):
                batch_start = time.clock()
#                 rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
                # cost = rbm.get_reconstruction_cross_entropy()
                # print >> sys.stderr, \
                #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost
                
                
                cost = 0.0;
                if batch_size == -1:
                    cost = rbm.contrastive_divergence(input = layer_input, lr=lr, k=k, batch_size = -1)
                else:
                    n_train_batches = len(layer_input) / batch_size # compute number of minibatches for training, validation and testing
                    mean_cost = []
                    for batch_index in xrange(n_train_batches):
                        mean_cost += [rbm.contrastive_divergence(input = layer_input [batch_index * batch_size:(batch_index + 1) * batch_size], lr=lr, k=k, batch_size = batch_size)]
                    cost = numpy.mean(mean_cost)

                
                batch_stop = time.clock()
                batch_time = (batch_stop - batch_start)
                
                print '\tPre-training layer [%d: %d X %d], epoch %d, cost %.7fm, entropy: %.2f, time is %.2f seconds' %(i, rbm.n_visible, rbm.n_hidden, epoch, cost, rbm.get_reconstruction_cross_entropy(), (batch_time))

            # synchronous betwen rbm and sigmoid layer
            self.sigmoid_layers[i].W = rbm.W
            self.sigmoid_layers[i].b = rbm.hbias

#                 # Plot filters after each training epoch
#                 # Construct image from the weight matrix
#                 if layer == 0:
#                     if (epoch % 20 == 0):
#                         image = PIL.Image.fromarray(tile_raster_images(
#                                  X = numpy.array(rbm.get_w().T),
#                                  img_shape=(28, 28), 
#                                  tile_shape=(10, 10),
#                                  tile_spacing=(1, 1)))
#                         image.save('result/filters_at_layer_%d_epoch_%d.png' % (layer, epoch))
            
#             numpy.array(rbm.get_w().T).dump(('result/weight_at_layer_%d.txt' % layer))
#             if layer == 0:
#                 image = PIL.Image.fromarray(tile_raster_images(
#                              X = numpy.array(rbm.get_w().T),
#                              img_shape=(28, rbm.n_visible / 28), 
#                              tile_shape=(10, 10),
#                              tile_spacing=(1, 1)))
#             image.save('result/filters_at_layer_%d.png' % layer)
        
        pretaining_end_time = time.clock()
        print ('Total time for pretraining: ' + '%.2f seconds' % ((pretaining_end_time - pretaining_start_time)))
        print self.sigmoid_layers[0].W

    def finetune(self, lr=0.1, epochs=100):
        
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()

#         self.log_layer = LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
#                                     label=self.y,
#                                     n_in=hidden_layer_sizes[-1],
#                                     n_out=n_outs)
                

        # train log_layer
        epoch = 0
        done_looping = False
        start_time = time.clock()
        print layer_input
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input=layer_input)
            if (epoch % 20 == 0):
                print ('\tFine-tuning epoch %d, cost is ' % epoch) #self.log_layer.negative_log_likelihood()
            
            lr *= 0.95
            epoch += 1
        end_time = time.clock()
        print ('Total time for fine-tuning: ' + '%.2f seconds' % ((end_time - start_time)))
        
        
        

    def predict(self, x = None, y = None):

        input_x = x
        
        for i in xrange(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            input_x = sigmoid_layer.output(input=input_x)
            
        

        out = self.log_layer.predict(input_x, y)
        return out





def test_dbn2(pretrain_lr=0.6, pretraining_epochs=300,
             k=1, finetune_lr=0.1, finetune_epochs=300,
             dataset='mnist.pkl.gz', batch_size=50, hidden_layer_sizes=[5, 5]):


    #########################
    # Data Loading
    #########################
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # cutting number of instances for training and testing
    n_train = 50000 # 50000
    n_test = 10000 # 10000
    train_set_x = numpy.array(train_set_x[0:n_train]) #  / 255.0
    train_set_y = numpy.array(train_set_y[0:n_train])
    test_set_x = numpy.array(test_set_x[0:n_test]) #  / 255.0
    test_set_y = numpy.array(test_set_y[0:n_test])
    
    train_set_y = numpy.array(ylabelToArray(train_set_y, 10))
    
    

    
    numpy_rng = numpy.random.RandomState(123) # numpy random generator

    #########################
    # DBN Start# construct the Deep Belief Network
    #########################
    print '... building the module: ', hidden_layer_sizes
    dbn = DBN(input=train_set_x, label=train_set_y, \
              n_ins=28 * 28, hidden_layer_sizes=hidden_layer_sizes, n_outs=10, numpy_rng=numpy_rng)
    #500, 1000, 100   # 1000, 1000, 1000
 
    #########################
    # PRETRAINING THE MODEL (DBNUnsupervisedPreTuning) # We are using CD-1 here
    #########################
    print '... getting the pretraining functions'
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs, batch_size=batch_size)


    ########################
    # FINETUNING THE MODEL (DBNSupervisedFineTuning) #
    ########################
    print '... finetuning the model'
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)

    ########################
    # PREDICTING TEST DATASET USING TRAINED MODEL #
    ########################
    print '... predicting test dataset'    
    print dbn.predict(x=test_set_x, y=test_set_y)

    


def test_dbn(pretrain_lr=0.6, pretraining_epochs=200, k=1, \
             finetune_lr=0.6, finetune_epochs=200):

    x = numpy.array([[1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0]])
    y = numpy.array([[1, 0],
                     [1, 0],
                     [1, 0],
                     [0, 1],
                     [0, 1],
                     [0, 1]])

    
    rng = numpy.random.RandomState(123)

    # construct DBN
    dbn = DBN(input=x, label=y, n_ins=6, hidden_layer_sizes=[10 ], n_outs=2, numpy_rng=rng)
    

    # pre-training (TrainUnsupervisedDBN)
    dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs, batch_size = -1)
    
#     fine-tuning (DBNSupervisedFineTuning)
    dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)
 
 
    # test
    x = numpy.array([[1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0]])
    y = numpy.array([0, 0, 0, 1, 1, 1])
    print dbn.predict(x, y)



if __name__ == "__main__":
    
#      test_dbn2(hidden_layer_sizes=[500, 1000])
    test_dbn()  
    sys.exit(2)
    if len(sys.argv) == 3:
        layerStr = sys.argv[1]
        layers = layerStr.split(".")
        layer_sizes = []
        for l in layers:
            layer_sizes.append(int(l))
        batch_size=int(sys.argv[2])
      
        print 'layer sizes: ', layer_sizes, ', batch size: ', batch_size
        #test_dbn()
        test_dbn2(hidden_layer_sizes=layer_sizes, batch_size=batch_size)
    elif len(sys.argv) == 1:
        layer_sizes = [500, 500, 2000, 30]
        batch_size = -1
        print 'layer sizes: ', layer_sizes, ', batch size: ', batch_size        
        test_dbn2(hidden_layer_sizes=layer_sizes, batch_size=batch_size)
    else:
        print("Usage: %s layers (e.g., 100.500.100) batchsize (e.g., 100 or -1)" % sys.argv[0])
        sys.exit(2)
