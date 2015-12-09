import lasagne
import h5py
import random
import numpy
import theano.tensor as T
import lasagne
import theano
import pickle
import os
import wget
import PIL, PIL.Image

if not os.path.exists('fonts.hdf5'):
    wget.download('https://s3.amazonaws.com/erikbern/fonts.hdf5')

f = h5py.File('fonts.hdf5', 'r')
data = f['fonts']
n, k = data.shape[0], data.shape[1]
wh = data.shape[2] * data.shape[3]

def iterate_minibatches():
    while True:
        batch_is = numpy.zeros((k, n), dtype=theano.config.floatX)
        batch_js = numpy.zeros((k, k), dtype=theano.config.floatX)
        i = random.randint(0, n-1)
        for z in xrange(k):
            batch_is[z][i] = 1
            batch_js[z][z] = 1

        yield batch_is, batch_js


def get_model(input_i, input_j):
    input_i = lasagne.layers.InputLayer(shape=(None, n), input_var=input_i)
    input_j = lasagne.layers.InputLayer(shape=(None, k), input_var=input_j)
    input_i_bottleneck = lasagne.layers.DenseLayer(input_i, 64)
    input_j_bottleneck = lasagne.layers.DenseLayer(input_j, 64)
    network = lasagne.layers.ConcatLayer([input_i_bottleneck, input_j_bottleneck])
    for i in xrange(3):
        network = lasagne.layers.DenseLayer(network, 1024)

    output = lasagne.layers.DenseLayer(network, wh, nonlinearity=lasagne.nonlinearities.sigmoid)
    return output


input_i = T.matrix('input_i')
input_j = T.matrix('input_j')
output = T.matrix('output')

network = get_model(input_i, input_j)
prediction = lasagne.layers.get_output(network)

if os.path.exists('model.pickle'):
    print 'loading model...'
    lasagne.layers.set_all_param_values(network, pickle.load(open('model.pickle')))

print 'compiling...'
run_fn = theano.function([input_i, input_j], prediction)

print 'generating...'
for input_i, input_j in iterate_minibatches():
    pred = run_fn(input_i, input_j).reshape((input_i.shape[0], 64, 64))

    img = PIL.Image.new('L', (8 * 64, 8 * 64))
    for z in xrange(k):
        x, y = z % 8, z // 8
        img_char = PIL.Image.fromarray(numpy.uint8(((1.0 - pred[z]) * 255)))
        img.paste(img_char, (x * 64, y * 64))

    img.save('font.png')

