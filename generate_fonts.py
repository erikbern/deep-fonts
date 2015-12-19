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
n = 56443

def generate_input(batch_size=128):
    while True:
        b = random.randint(0, n-1)
        print b, '...'
        batch_is = numpy.zeros((k,), dtype=numpy.int32)
        batch_js = numpy.zeros((k,), dtype=numpy.int32)
        for z in xrange(k):
            batch_is[z] = b
            batch_js[z] = z

        yield batch_is, batch_js


model = model.Model(n, k, wh)
model.try_load()
run_fn = model.get_run_fn()

print 'generating...'
frame = 0
for input_i, input_j in generate_input():
    pred = run_fn(input_i, input_j).reshape((input_i.shape[0], 64, 64))

    img = PIL.Image.new('L', (8 * 64, 8 * 64), 255)
    for z in xrange(k):
        x, y = z % 8, z // 8
        img_char = PIL.Image.fromarray(numpy.uint8(((1.0 - pred[z]) * 255)))
        img.paste(img_char, (x * 64, y * 64))

    img.save('font_%06d.png' % frame)
    frame += 1


