import random
import numpy
import theano
import PIL, PIL.Image
import model

model = model.Model(artificial_font=True)
model.try_load()
run_fn = model.get_run_fn()
W = model.get_font_embeddings()

a, b, c, d = [random.choice(W) for z in xrange(4)]
grid = 10

batch_is = numpy.zeros((grid*grid, model.d), dtype=theano.config.floatX)
batch_js = numpy.zeros((grid*grid,), dtype=numpy.int32)
for x in xrange(grid):
    for y in xrange(grid):
        z = y * grid + x
        p, q = 1.0 * x / (grid-1), 1.0 * y / (grid-1)
        batch_is[z] = p * q * a + (1 - p) * q * b + p * (1 - q) * c + (1 - p) * (1 - q) * d
        batch_js[z] = 26 # lowercase a

pred = run_fn(batch_is, batch_js).reshape((grid * grid, 64, 64))

img = PIL.Image.new('L', (grid * 64, grid * 64), 255)
for x in xrange(grid):
    for y in xrange(grid):
        z = y * grid + x
        img_char = PIL.Image.fromarray(numpy.uint8(((1.0 - pred[z]) * 255)))
        img.paste(img_char, (x * 64, y * 64))

img.save('grid.png')



