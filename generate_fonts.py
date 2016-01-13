import random
import numpy
import theano
import model

m = model.Model(artificial_font=True)
m.try_load()
run_fn = m.get_run_fn()
W = m.get_font_embeddings()
cov = numpy.cov(W.T)

def generate_font():
    return numpy.random.multivariate_normal(mean=numpy.zeros(m.d), cov=cov)

def generate_input(n_fonts=5):
    fonts = [generate_font() for f in xrange(n_fonts)]
    for f in xrange(n_fonts):
        a, b = fonts[f], fonts[(f+1)%n_fonts]
        for p in numpy.linspace(0, 1, 10):
            print f, p
            batch_is = numpy.zeros((m.k, m.d), dtype=theano.config.floatX)
            batch_js = numpy.zeros((m.k,), dtype=numpy.int32)
            for z in xrange(m.k):
                batch_is[z] = a * (1-p) + b * p
                batch_js[z] = z

            yield batch_is, batch_js

print 'generating...'
frame = 0
for input_i, input_j in generate_input():
    img = model.draw_grid(run_fn(input_i, input_j))
    img.save('font_%06d.png' % frame)
    frame += 1
