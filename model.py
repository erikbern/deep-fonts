import lasagne
import theano
import theano.tensor as T
import os
import pickle

class Model(object):
    def __init__(self, n, k, wh):
        self.input_i = T.matrix('input_i')
        self.input_j = T.matrix('input_j')
        self.target = T.matrix('target')
        
        input_i = lasagne.layers.InputLayer(shape=(None, n), input_var=self.input_i)
        input_j = lasagne.layers.InputLayer(shape=(None, k), input_var=self.input_j)
        input_i_bottleneck = lasagne.layers.DenseLayer(input_i, 64)
        input_j_bottleneck = lasagne.layers.DenseLayer(input_j, 64)
        network = lasagne.layers.ConcatLayer([input_i_bottleneck, input_j_bottleneck])
        for i in xrange(3):
            network = lasagne.layers.DenseLayer(network, 1024)

        network = lasagne.layers.DenseLayer(network, wh, nonlinearity=lasagne.nonlinearities.sigmoid)
        self.network = network
        self.prediction = lasagne.layers.get_output(network)
        print self.prediction.dtype

    def get_train_fn(self):
        print 'compiling training fn'
        loss = lasagne.objectives.squared_error(self.prediction, self.target).mean()
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=lasagne.utils.floatX(1.0), momentum=lasagne.utils.floatX(0.9))

        return theano.function([self.input_i, self.input_j, self.target], loss, updates=updates)

    def get_run_fn(self):
        return theano.function([self.input_i, self.input_j], self.prediction)

    def try_load(self):
        if os.path.exists('model.pickle'):
            print 'loading model...'
            lasagne.layers.set_all_param_values(self.network, pickle.load(open('model.pickle')))

    def save(self):
        print 'saving model...'
        params = lasagne.layers.get_all_param_values(self.network)
        f = open('model.pickle', 'w')
        pickle.dump(params, f)
        f.close()
