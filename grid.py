import random
import numpy
import theano
import model

m = model.Model(artificial_font=True)
m.try_load()
run_fn = m.get_run_fn()
W = m.get_font_embeddings()

a, b, c, d = [random.choice(W) for z in xrange(4)]
grid = 10

batch_is = numpy.zeros((grid*grid, m.d), dtype=theano.config.floatX)
batch_js = numpy.zeros((grid*grid,), dtype=numpy.int32)
for x in xrange(grid):
    for y in xrange(grid):
        z = y * grid + x
        p, q = 1.0 * x / (grid-1), 1.0 * y / (grid-1)
        batch_is[z] = p * q * a + (1 - p) * q * b + p * (1 - q) * c + (1 - p) * (1 - q) * d
        batch_js[z] = 26 # lowercase a

pred = run_fn(batch_is, batch_js)
model.draw_grid(pred, grid).save('grid.png')
