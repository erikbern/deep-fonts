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


def loss(a, b):
    return 0.5 * abs(a-b) + 0.5 * (a - b)**2


class Model(object):
    def __init__(self, n, k, wh, d=64, lambd=1e-8, artificial_font=False):
        self.target = T.matrix('target')

        if artificial_font:
            self.input_font = T.matrix('input_font')
            input_font_bottleneck = lasagne.layers.InputLayer(shape=(None, d), input_var=self.input_font, name='input_font_emb')
        else:
            self.input_font = T.ivector('input_font')
            input_font = lasagne.layers.InputLayer(shape=(None,), input_var=self.input_font, name='input_font')
            input_font_one_hot = OneHotLayer(input_font, n)
            input_font_bottleneck = lasagne.layers.DenseLayer(input_font_one_hot, d, name='input_font_bottleneck', b=None, nonlinearity=None)

        self.input_char = T.ivector('input_char')
        input_char = lasagne.layers.InputLayer(shape=(None,), input_var=self.input_char, name='input_char')
        input_char_one_hot = OneHotLayer(input_char, k)
        input_char_bottleneck = lasagne.layers.DenseLayer(input_char_one_hot, d, name='input_char_bottleneck', b=None, nonlinearity=None)

        network = lasagne.layers.ConcatLayer([input_font_bottleneck, input_char_bottleneck], name='input_concat')
        network = lasagne.layers.DropoutLayer(network, name='input_concat_dropout')
        for i in xrange(4):
            network = lasagne.layers.DenseLayer(network, 2048, name='dense_%d' % i)

        network = lasagne.layers.DenseLayer(network, wh, nonlinearity=self.last_nonlinearity, name='output_sigmoid')
        self.network = network
        self.prediction_train = lasagne.layers.get_output(network)
        self.prediction = lasagne.layers.get_output(network, deterministic=True)
        print self.prediction.dtype
        self.loss = loss(self.prediction_train, self.target).mean()
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
        values = pickle.load(open('model.pickle'))
        for p in lasagne.layers.get_all_params(self.network):
            if p.name not in values:
                print 'dont have value for', p.name
            else:
                value = values[p.name]
                if p.get_value().shape != value.shape:
                    print p.name, ':', p.get_value().shape, 'and', value.shape, 'have different shape!!!'
                else:
                    p.set_value(value.astype(theano.config.floatX))

    def save(self):
        print 'saving model...'
        params = {}
        for p in lasagne.layers.get_all_params(self.network):
            params[p.name] = p.get_value()
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
