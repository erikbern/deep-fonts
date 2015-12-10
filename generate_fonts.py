import h5py
import random
import numpy
import theano
import pickle
import os
import wget
import PIL, PIL.Image
import model

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

model = model.Model(n, k, wh)
model.try_load()
run_fn = model.get_run_fn()

print 'generating...'
for input_i, input_j in iterate_minibatches():
    pred = run_fn(input_i, input_j).reshape((input_i.shape[0], 64, 64))

    img = PIL.Image.new('L', (8 * 64, 8 * 64))
    for z in xrange(k):
        x, y = z % 8, z // 8
        img_char = PIL.Image.fromarray(numpy.uint8(((1.0 - pred[z]) * 255)))
        img.paste(img_char, (x * 64, y * 64))

    img.save('font.png')

