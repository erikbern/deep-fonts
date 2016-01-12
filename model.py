import lasagne
import theano
import theano.tensor as T
import os
import pickle
import numpy
import h5py
import wget
from sklearn import cross_validation

class OneHotLayer(lasagne.layers.Layer):
    def __init__(self, incoming, nb_class, **kwargs):
        super(OneHotLayer, self).__init__(incoming, **kwargs)
        self.nb_class = nb_class

    def get_output_for(self, incoming, **kwargs):
        return theano.tensor.extra_ops.to_one_hot(incoming, self.nb_class)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nb_class)


def loss(a, b):
    # return 0.5 * abs(a-b) + 0.5 * (a - b)**2
    return abs(a-b)


class Model(object):
    def __init__(self, n=None, k=62, wh=64*64, d=40, D=1024, lambd=1e-7, artificial_font=False):
        self.n, self.k, self.d = n, k, d
        self.target = T.matrix('target')

        if artificial_font:
            self.input_font = T.matrix('input_font')
            input_font_bottleneck = lasagne.layers.InputLayer(shape=(None, d), input_var=self.input_font, name='input_font_emb')
        else:
            self.input_font = T.ivector('input_font')
            input_font = lasagne.layers.InputLayer(shape=(None,), input_var=self.input_font, name='input_font')
            input_font_one_hot = OneHotLayer(input_font, n)
            input_font_bottleneck = lasagne.layers.DenseLayer(input_font_one_hot, d, name='input_font_bottleneck', nonlinearity=None, b=None)

        self.input_char = T.ivector('input_char')
        input_char = lasagne.layers.InputLayer(shape=(None,), input_var=self.input_char, name='input_char')
        input_char_one_hot = OneHotLayer(input_char, k)

        network = lasagne.layers.ConcatLayer([input_font_bottleneck, input_char_one_hot], name='input_concat')
        for i in xrange(4):
            network = lasagne.layers.DenseLayer(network, D, name='dense_%d' % i, nonlinearity=lasagne.nonlinearities.leaky_rectify)

        network = lasagne.layers.DenseLayer(network, wh, nonlinearity=lasagne.nonlinearities.sigmoid, name='output_sigmoid')
        self.network = network
        self.prediction_train = lasagne.layers.get_output(network)
        self.prediction = lasagne.layers.get_output(network, deterministic=True)
        print self.prediction.dtype
        self.loss = loss(self.prediction_train, self.target).mean()
        self.reg = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2) * lambd
        self.input_font_bottleneck = input_font_bottleneck

    def get_train_fn(self):
        print 'compiling training fn'
        learning_rate = T.scalar('learning_rate')
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(self.loss + self.reg, params, learning_rate=learning_rate, momentum=lasagne.utils.floatX(0.9))
        return theano.function([learning_rate, self.input_font, self.input_char, self.target], [self.loss, self.reg], updates=updates)

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
        data = pickle.load(open('model.pickle'))
        return data['input_font_bottleneck.W']

    def sets(self):
        dataset = []
        for i in xrange(self.n):
            for j in xrange(self.k):
                dataset.append((i, j))

        train_set, test_set = cross_validation.train_test_split(dataset, test_size=0.10, random_state=0)
        return train_set, test_set

def get_data():
    if not os.path.exists('fonts.hdf5'):
        wget.download('https://s3.amazonaws.com/erikbern/fonts.hdf5')

    f = h5py.File('fonts.hdf5', 'r')
    return f['fonts']
