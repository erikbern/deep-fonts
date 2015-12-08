import lasagne
import h5py
import random
import numpy
import theano.tensor as T
import lasagne
import theano
from matplotlib import pyplot
import pickle
import os

f = h5py.File('fonts.hdf5', 'r')
data = f['fonts']
n, k = data.shape[0], data.shape[1]
wh = data.shape[2] * data.shape[3]


def iterate_minibatches(batch_size=128):
    while True:
        batch_is = numpy.zeros((batch_size, n), dtype=numpy.int32)
        batch_js = numpy.zeros((batch_size, k), dtype=numpy.int32)
        batch_ds = numpy.zeros((batch_size, wh))
        for z in xrange(batch_size):
            i = random.randint(0, n-1)
            j = random.randint(0, k-1)
            batch_is[z][i] = 1
            batch_is[z][j] = 1
            batch_ds[z] = data[i][j].flatten() * 1. / 255

        yield batch_is, batch_js, batch_ds


def get_model(input_i, input_j):
    input_i = lasagne.layers.InputLayer(shape=(None, n), input_var=input_i)
    input_j = lasagne.layers.InputLayer(shape=(None, k), input_var=input_j)
    network = lasagne.layers.ConcatLayer([input_i, input_j])
    for i in xrange(3):
        network = lasagne.layers.DenseLayer(network, 1024)

    output = lasagne.layers.DenseLayer(network, wh, nonlinearity=lasagne.nonlinearities.sigmoid)
    return output


input_i = T.imatrix('input_i')
input_j = T.imatrix('input_j')
output = T.matrix('output')

network = get_model(input_i, input_j)
prediction = lasagne.layers.get_output(network)
print prediction.dtype
loss = lasagne.objectives.binary_crossentropy(prediction, output).mean()
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=lasagne.utils.floatX(0.1), momentum=lasagne.utils.floatX(0.9))

if os.path.exists('model.pickle'):
    print 'loading model...'
    lasagne.layers.set_all_param_values(network, pickle.load(open('model.pickle')))

print 'compiling...'
train_fn = theano.function([input_i, input_j, output], loss, updates=updates)
run_fn = theano.function([input_i, input_j], prediction)

print 'training...'
for input_i, input_j, output in iterate_minibatches():
    print train_fn(input_i, input_j, output)
    real = output.reshape(output.shape[0], 64, 64)
    pred = run_fn(input_i, input_j).reshape((output.shape[0], 64, 64))
    if random.random() < 0.01:
        print 'saving model...'
        params = lasagne.layers.get_all_param_values(network)
        f = open('model.pickle', 'w')
        pickle.dump(params, f)
        f.close()
        
        f, (ax1, ax2) = pyplot.subplots(1, 2)
        ax1.matshow(real[0], cmap='gray')
        ax2.matshow(pred[0], cmap='gray')
        f.savefig("real_vs_pred.png")
        pyplot.clf()
