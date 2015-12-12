import h5py
import random
import numpy
import theano
from matplotlib import pyplot
import pickle
import os
import wget
import model
from sklearn import cross_validation
import sys

if not os.path.exists('fonts.hdf5'):
    wget.download('https://s3.amazonaws.com/erikbern/fonts.hdf5')

f = h5py.File('fonts.hdf5', 'r')
data = f['fonts']
n, k = data.shape[0], data.shape[1]
wh = data.shape[2] * data.shape[3]

dataset = []
for i in xrange(n):
    for j in xrange(k):
        dataset.append((i, j))

train_set, test_set = cross_validation.train_test_split(dataset, test_size=2000, random_state=0)

def iterate_minibatches(dataset, batch_size=128):
    random.shuffle(dataset)
    for offset in xrange(0, len(dataset) - batch_size, batch_size):
        batch_fonts = numpy.zeros((batch_size, n), dtype=theano.config.floatX)
        batch_chars = numpy.zeros((batch_size, k), dtype=theano.config.floatX)
        batch_ds = numpy.zeros((batch_size, wh), dtype=theano.config.floatX)
        for z in xrange(batch_size):
            i, j = dataset[offset + z]
            batch_fonts[z][i] = 1
            batch_chars[z][j] = 1
            batch_ds[z] = data[i][j].flatten() * 1. / 255

        yield batch_fonts, batch_chars, batch_ds

model = model.Model(n, k, wh)
model.try_load()
train_fn = model.get_train_fn(updates=True)
test_fn = model.get_train_fn(updates=False)
run_fn = model.get_run_fn()

print 'training...'
epoch = 0
while True:
    print 'epoch', epoch
    for input_font, input_char, output in iterate_minibatches(train_set):
        loss, reg = train_fn(input_font, input_char, output)
        sys.stdout.write('Train perf: %.9f %.9f\r' % (float(loss), float(reg)))
        sys.stdout.flush()
        break
    sys.stdout.write('\n')
    epoch += 1
    total_loss, total_reg = 0, 0
    for input_Font, input_char, output in iterate_minibatches(test_set):
        loss, reg = test_fn(input_font, input_char, output)
        total_loss += loss
        total_reg += reg
    sys.stdout.write('Test perf: %.9f %.9f\n' % (float(loss), float(reg)))
    model.save()
    
    continue
    real = output.reshape(output.shape[0], 64, 64)
    pred = run_fn(input_font, input_char).reshape((output.shape[0], 64, 64))
    f, (ax1, ax2) = pyplot.subplots(1, 2)
    ax1.matshow(real[0], cmap='gray')
    ax2.matshow(pred[0], cmap='gray')
    f.savefig("real_vs_pred.png")
    pyplot.clf()
