
import time
import sys
import numpy 
from numpy import mean, sqrt
import math
from utils import softmax, sigmoid
from utils import *


import PIL.Image
#from matplotlib.pyplot import hist, title, subplot
#from pylab import *
# import theano
# from theano import tensor as T


class RBM(object):

    def __init__(self, input=None, n_visible=2, n_hidden=3, W=None, hbias=None, vbias=None, numpy_rng=None):
        
        
        
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden  # num of units in hidden layer

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / (n_visible * n_hidden)
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))
            W = 0.1 * initial_W
            
#                     scale = 0.001
#             initial_W = 2 * scale * numpy.random.normal(0, 0.1, (n_visible, n_hidden))

            
#             W = (initial_W  * 2 - 1) * 0.1          
#             initial_W = numpy.array(numpy.random.uniform(
#                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
#                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
#                    size=(n_visible, n_hidden)))           
#             W = initial_W


        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0
            

        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        
#         self.momentum_speed =  numpy.zeros(shape=(n_visible,n_hidden))
        
        self.Wu = numpy.zeros(shape=(n_visible,n_hidden))
        self.hu = numpy.zeros(shape=(n_hidden))
        self.vu = numpy.zeros(shape=(n_visible))



        self.params = [self.W, self.hbias, self.vbias]


        
        
    def contrastive_divergence(self, input=None, lr=0.1, k=1, batch_size = -1):  
#         if input is not None:
#             self.input = input
                
                

#         print self.input.shape
#         print self.input
        
        
        ''' CD-k '''
         # v -> h0 -> v1, h1
        h0_mean, h0_sample = self.sample_h_given_v(input)
#         chain_start = h0_sample

        for step in xrange(k):
            if step == 0:
                v1_means, v1_samples, h1_means, h1_samples = self.gibbs_hvh(h0_sample)
            else:
                v1_means, v1_samples, h1_means, h1_samples = self.gibbs_hvh(h1_samples)

        
       
        
        # Weight Averaging Function
        # Update the weights with the difference in correlations between the positive and negative phases.
        gW = (numpy.dot(input.T, h0_sample) - numpy.dot(v1_samples.T, h1_means))   # (784,500) = [784][50000] * [500][50000] - [784][50000] * [500][50000]
        # The gradient of the visible biases is the difference in summed visible activities for the minibatch.
        gvbias = numpy.mean(input - v1_samples, axis=0)                  # (784) = [50000][784] - [50000][784]     
        #  The gradient of the hidden biases is the difference in summed hidden activities for the minibatch.     
        ghbias = numpy.mean(h0_sample - h1_means, axis=0)                   # (500) = [50000][500] - [50000][500]          
        
        
#         cost = numpy.mean(self.free_energy(self.input), axis=0) - numpy.mean(self.free_energy(v1_samples), axis=0)
#         
# 
#         
#         gparams = T.grad(cost, params_theano, consider_constant=[v1_samples], dtype=theano.config.floatX)
#         print gparams
        
        
#         # constructs the update dictionary
#         for gparam, param in zip(gparams, self.params):
#             # make sure that the learning rate is of the right dtype
#             updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
        
        
        if batch_size == -1:
            self.W += lr * gW / len(self.input)
            self.vbias += lr * gvbias / len(self.input)
            self.hbias += lr * ghbias / len(self.input)
        else:
            momentum_weight= 0.1 # 0.0001
            l2=0.9 
            momentum=0.2 #0.9
            batch_size = len(input)
#             self.W = self.W + momentum * self.momentum_speed + momentum_weight * (lr * gW / batch_size - 0.0002 * self.W)
#             self.vbias = self.vbias + momentum_weight * lr * gvbias / batch_size       
#             self.hbias = self.hbias + momentum_weight * lr * ghbias / batch_size
#             self.W = self.W + lr * ( (1 - momentum) * gW / batch_size + momentum * ( gW - l2 * self.W))
#             self.vbias = self.vbias + lr * ( (1 - momentum)  * gvbias / batch_size + momentum  * ( gvbias * self.vbias) )
#             self.hbias = self.hbias + lr * ( (1 - momentum)  * ghbias / batch_size + momentum  * ( ghbias * self.hbias) )
            
            
#             batch_size = 1
#             self.W += lr * gW / batch_size
#             self.vbias += lr * gvbias / batch_size
#             self.hbias += lr * ghbias / batch_size
            
#             momentum=0.9
#             batch_size = len(input)
#             self.Wu = self.Wu * momentum + gW
#             self.vu = self.vu * momentum + gvbias
#             self.hu = self.hu * momentum + ghbias
#             

            
#             self.W = self.W + lr * self.Wu / len(self.input)
#             self.vbias = self.vbias + lr * self.vu / len(self.input)
#             self.hbias = self.hbias + lr * self.hu / len(self.input)

            
#             self.Wu = 0.7 * self.Wu + lr * ( gW / batch_size - 0.0002 * self.W )
#             self.hu = 0.7 * self.hbias + (lr / batch_size) * ghbias
#             self.vu = 0.7 * self.vbias + (lr / batch_size) * gvbias
#             
#             self.W = self.W + self.Wu
#             self.hbias = self.hbias + self.hu
#             self.vbias = self.vbias + self.vu

           
#             self.momentum_speed = self.momentum_speed * 0.9 + 0.0001 * (gW / batch_size - 0.0002 * self.W)
#             self.W = self.W + self.momentum_speed
#             self.vbias = self.vbias + 0.0001 * (gvbias / batch_size)
#             self.hbias = self.hbias + 0.0001 * (ghbias / batch_size)
            
            
                
        
        
#         print self.W

        return sqrt(mean(( gvbias / len(self.input) )**2))

          
    """
    Contrastive Divergence Functions
    """  
    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample, h1_mean, h1_sample]
                
    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.numpy_rng.binomial(size=h1_mean.shape, n=1,p=h1_mean)
        return [h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.numpy_rng.binomial(size=v1_mean.shape, n=1,p=v1_mean)
        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = numpy.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)



    """
    Entropy and Energy Functions
    """
    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        
        pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy = -numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v) + 
            (1 - self.input) * numpy.log(1 - sigmoid_activation_v),  axis=1))
        
        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(numpy.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(numpy.dot(h, self.W.T) + self.vbias)
        return reconstructed_v

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        cross_entropy = T.mean(
                T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) + 
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)), axis=1))
        return cross_entropy    

    def free_energy(self, v_sample):
        wx_b = numpy.dot(v_sample, self.W) + self.hbias
        vbias_term = numpy.dot(v_sample, self.vbias)
        hidden_term = numpy.sum(numpy.log(1 + numpy.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term
    



    def get_w(self):
        return self.W;


def test_rbm2(learning_rate=0.1, training_epochs=20,
             dataset='mnist.pkl.gz', batch_size=30,
             n_chains=20, n_samples=10, output_folder='result',
             n_hidden=500, k=3):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_ins = 5000
    train_set_x = numpy.array(train_set_x[0:n_ins])
    train_set_y = numpy.array(train_set_y[0:n_ins])

    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_set_x) / batch_size  # .get_value(borrow=True).shape[0] 
    rng = numpy.random.RandomState(123)



    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    

    plotting_time = 0.
    start_time = time.clock()
    
    # construct RBM
    rbm = RBM(n_visible=28 * 28, n_hidden=n_hidden, numpy_rng=rng, input = train_set_x)

    # go through training epochs
    for epoch in xrange(training_epochs):


        batch_start = time.clock()
        # Full-Batch
#         mean_cost = rbm.contrastive_divergence(lr=learning_rate, k=k)
        
        # Mini-Batch
        mean_cost = []
        for batch_index in xrange(n_train_batches):
#             print 'batch: ', batch_index, ' shape:', train_set_x [batch_index * batch_size:(batch_index + 1) * batch_size].shape
            mean_cost += [rbm.contrastive_divergence(input = train_set_x [batch_index * batch_size:(batch_index + 1) * batch_size], lr=learning_rate, k=k)]
#             print mean_cost

        batch_stop = time.clock()
        batch_time = (batch_stop - batch_start)
        plotting_time += (batch_time)

        print >> sys.stderr, \
        'Training epoch %d, cost is %.7f, time is %.2f seconds, entropy: %.2f, energy: %.2f ' % (epoch, numpy.mean(mean_cost), (batch_time), rbm.get_reconstruction_cross_entropy(), numpy.mean(rbm.free_energy(train_set_x)))



        if(epoch % 5 == 0):
            
            hMean = sigmoid(numpy.dot(rbm.input, rbm.W) + rbm.hbias)
            normhMean = ((hMean - hMean.min()) / (hMean.max() - hMean.min() + 1e-6))
            image = PIL.Image.fromarray(normhMean * 256)
            image.convert('RGB').save('hmean_at_epoch_%i.jpg' % epoch)

            
            image2 = PIL.Image.fromarray(tile_raster_images(
                     X=numpy.array(rbm.W.T),
                     img_shape=(28, 28), tile_shape=(10, 10),
                     tile_spacing=(1, 1)))
            image2.convert('RGB').save('filters_at_epoch_%i.jpg' % epoch)

            

#             fig = figure()
#             subplot(231); plotit(rbm.vbias)
#             subplot(232); plotit(rbm.W.flatten())
#             subplot(233); plotit(rbm.hbias)
#             subplot(234); plotit(rbm.vbias)
#             subplot(235); plotit(rbm.W.flatten())
#             subplot(236); plotit(rbm.hbias)
# #             subplot_tool()
#             show()



    end_time = time.clock()
    pretraining_time = (end_time - start_time) - plotting_time
    print ('Training took %f seconds' % (pretraining_time))

    #################################
    #     Sampling from the RBM     #
    #################################
    reconstructed_x = rbm.reconstruct(test_set_x)
    img3 = PIL.Image.fromarray(tile_raster_images(X=reconstructed_x, img_shape=(28,28), tile_shape=(100,100), tile_spacing=(1,1)))
    img3.convert('RGB').save('reconstructed_v.jpg')

#     print reconstructed_x
    # test images using reconstrcuted[10,000][100]



def test_rbm(learning_rate=0.1, k=1, training_epochs=1000):
    data = numpy.array([[1, 1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0]])

    rng = numpy.random.RandomState(123)

    # construct RBM
    rbm = RBM(input=data, n_visible=6, n_hidden=2, numpy_rng=rng)

    # train
    for epoch in xrange(training_epochs):
        cost = rbm.contrastive_divergence(input=data, lr=learning_rate, k=k)
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost


    # test
    v = numpy.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0]])

    print rbm.reconstruct(v)

#def plotit(values):
    #hist(values);
    #title('mm = %g' % mean(fabs(values)))






if __name__ == "__main__":
    test_rbm2()
    #test_rbm2()




