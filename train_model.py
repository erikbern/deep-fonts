import h5py
import random
import numpy
import theano
from matplotlib import pyplot
import pickle
import os
import wget
import model

if not os.path.exists('fonts.hdf5'):
    wget.download('https://s3.amazonaws.com/erikbern/fonts.hdf5')

f = h5py.File('fonts.hdf5', 'r')
data = f['fonts']
n, k = data.shape[0], data.shape[1]
wh = data.shape[2] * data.shape[3]


def iterate_minibatches(batch_size=128):
    while True:
        batch_is = numpy.zeros((batch_size, n), dtype=theano.config.floatX)
        batch_js = numpy.zeros((batch_size, k), dtype=theano.config.floatX)
        batch_ds = numpy.zeros((batch_size, wh), dtype=theano.config.floatX)
        for z in xrange(batch_size):
            i = random.randint(0, n-1)
            j = random.randint(0, k-1)
            batch_is[z][i] = 1
            batch_js[z][j] = 1
            batch_ds[z] = data[i][j].flatten() * 1. / 255

        yield batch_is, batch_js, batch_ds

model = model.Model(n, k, wh)
model.try_load()
train_fn = model.get_train_fn()
run_fn = model.get_run_fn()

print 'training...'
for input_i, input_j, output in iterate_minibatches():
    print train_fn(input_i, input_j, output)
    real = output.reshape(output.shape[0], 64, 64)
    if random.random() < 0.001:
        model.save()

    pred = run_fn(input_i, input_j).reshape((output.shape[0], 64, 64))
    f, (ax1, ax2) = pyplot.subplots(1, 2)
    ax1.matshow(real[0], cmap='gray')
    ax2.matshow(pred[0], cmap='gray')
    f.savefig("real_vs_pred.png")
    pyplot.clf()
