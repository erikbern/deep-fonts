import lasagne
import theano
import theano.tensor as T
import os
import pickle

class Model(object):
    def __init__(self, n, k, wh):
        self.input_font = T.matrix('input_font')
        self.input_char = T.matrix('input_char')
        self.target = T.matrix('target')
        
        input_font = lasagne.layers.InputLayer(shape=(None, n), input_var=self.input_font)
        input_char = lasagne.layers.InputLayer(shape=(None, k), input_var=self.input_char)
        input_font_bottleneck = lasagne.layers.DenseLayer(input_font, 256)
        input_char_bottleneck = lasagne.layers.DenseLayer(input_char, 64)
        network = lasagne.layers.ConcatLayer([input_font_bottleneck, input_char_bottleneck])
        network = lasagne.layers.DropoutLayer(network)
        for i in xrange(4):
            network = lasagne.layers.DenseLayer(network, 2048)

        network = lasagne.layers.DenseLayer(network, wh, nonlinearity=lasagne.nonlinearities.sigmoid)
        self.network = network
        self.prediction_train = lasagne.layers.get_output(network)
        self.prediction = lasagne.layers.get_output(network, deterministic=True)
        print self.prediction.dtype

    def get_train_fn(self, lambd=1e-7):
        print 'compiling training fn'
        loss = lasagne.objectives.squared_error(self.prediction_train, self.target).mean()
        reg = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2) * lambd
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=lasagne.utils.floatX(1.0), momentum=lasagne.utils.floatX(0.9))

        return theano.function([self.input_font, self.input_char, self.target], [loss, reg], updates=updates)

    def get_run_fn(self):
        return theano.function([self.input_font, self.input_char], self.prediction)

    def try_load(self):
        if not os.path.exists('model.pickle'):
            return
        print 'loading model...'
        params = lasagne.layers.get_all_params(self.network)
        values = pickle.load(open('model.pickle'))
        for p, v in zip(params, values):
            if p.get_value().shape != v.shape:
                print p, p.get_value().shape, 'and', v.shape, 'have different shape!!!'
            else:
                p.set_value(v)

    def save(self):
        print 'saving model...'
        params = lasagne.layers.get_all_param_values(self.network)
        f = open('model.pickle', 'w')
        pickle.dump(params, f)
        f.close()
