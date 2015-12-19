import lasagne
import theano
import theano.tensor as T
import os
import pickle
import numpy

class OneHotLayer(lasagne.layers.Layer):
    def __init__(self, incoming, nb_class, **kwargs):
        super(OneHotLayer, self).__init__(incoming, **kwargs)
        self.nb_class = nb_class

    def get_output_for(self, incoming, **kwargs):
        return theano.tensor.extra_ops.to_one_hot(incoming, self.nb_class)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nb_class)


def absolute_error(a, b):
    return abs(a-b)


class Model(object):
    def __init__(self, n, k, wh, lambd=1e-8):
        self.input_font = T.ivector('input_font')
        self.input_char = T.ivector('input_char')
        self.target = T.matrix('target')
        
        input_font = lasagne.layers.InputLayer(shape=(None,), input_var=self.input_font, name='input_font')
        input_char = lasagne.layers.InputLayer(shape=(None,), input_var=self.input_char, name='input_char')
        input_font_one_hot = OneHotLayer(input_font, n)
        input_char_one_hot = OneHotLayer(input_char, k)
        input_font_bottleneck = lasagne.layers.DenseLayer(input_font_one_hot, 64, name='input_font_bottleneck', b=None, nonlinearity=None)
        input_char_bottleneck = lasagne.layers.DenseLayer(input_char_one_hot, 64, name='input_char_bottleneck', b=None, nonlinearity=None)
        network = lasagne.layers.ConcatLayer([input_font_bottleneck, input_char_bottleneck], name='input_concat')
        network = lasagne.layers.DropoutLayer(network, name='input_concat_dropout')
        for i in xrange(4):
            network = lasagne.layers.DenseLayer(network, 2048, name='dense_%d' % i)

        network = lasagne.layers.DenseLayer(network, wh, nonlinearity=self.last_nonlinearity, name='output_sigmoid')
        self.network = network
        self.prediction_train = lasagne.layers.get_output(network)
        self.prediction = lasagne.layers.get_output(network, deterministic=True)
        print self.prediction.dtype
        # self.loss = lasagne.objectives.squared_error(self.prediction_train, self.target).mean()
        self.loss = absolute_error(self.prediction_train, self.target).mean()
        self.reg = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2) * lambd
        self.input_font_bottleneck = input_font_bottleneck

    def get_train_fn(self):
        print 'compiling training fn'
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(self.loss + self.reg, params, learning_rate=lasagne.utils.floatX(1.0), momentum=lasagne.utils.floatX(0.9))
        return theano.function([self.input_font, self.input_char, self.target], [self.loss, self.reg], updates=updates)

    def get_test_fn(self):
        print 'compiling testing fn'
        params = lasagne.layers.get_all_params(self.network, trainable=False)
        return theano.function([self.input_font, self.input_char, self.target], [self.loss, self.reg])

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

    def get_font_embeddings(self):
        return self.input_font_bottleneck.W.get_value()

    def last_nonlinearity(self, x, T=4.0):
        return theano.tensor.nnet.sigmoid(x)
        #mean = theano.tensor.mean(x, axis=1, keepdims=True)
        #std = theano.tensor.std(x, axis=1, keepdims=True)
        #z = mean + T * (x - mean) / std # extremize
        #return theano.tensor.nnet.sigmoid(z)
