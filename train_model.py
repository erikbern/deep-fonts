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

train_set, test_set = cross_validation.train_test_split(dataset, test_size=10000, random_state=0)

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

        
def iterate_run(dataset, fn, tag):
    total_loss, total_reg, total_count = 0, 0, 0
    for input_font, input_char, output in iterate_minibatches(dataset):
        loss, reg = fn(input_font, input_char, output)
        total_loss += float(loss)
        total_reg += float(reg)
        total_count += 1
        sys.stdout.write('%s perf: %.9f %.9f accumulated: %.9f %.9f\r' % (tag, float(loss), float(reg), total_loss / total_count, total_reg / total_count))
        sys.stdout.flush()

    sys.stdout.write('\n')


model = model.Model(n, k, wh)
model.try_load()
train_fn = model.get_train_fn(updates=True)
test_fn = model.get_train_fn(updates=False)
run_fn = model.get_run_fn()

print 'training...'
epoch = 0
while True:
    print 'epoch', epoch
    iterate_run(train_set, train_fn, 'train')
    iterate_run(test_set, test_fn, 'test ')
    epoch += 1
    model.save()

    continue
    real = output.reshape(output.shape[0], 64, 64)
    pred = run_fn(input_font, input_char).reshape((output.shape[0], 64, 64))
    f, (ax1, ax2) = pyplot.subplots(1, 2)
    ax1.matshow(real[0], cmap='gray')
    ax2.matshow(pred[0], cmap='gray')
    f.savefig("real_vs_pred.png")
    pyplot.clf()
