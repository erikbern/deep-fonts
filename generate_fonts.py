import random
import numpy
import theano
import PIL, PIL.Image
import model

model = model.Model(artificial_font=True)
model.try_load()
run_fn = model.get_run_fn()
W = model.get_font_embeddings()
cov = numpy.cov(W.T)

def generate_font():
    return numpy.random.multivariate_normal(mean=numpy.zeros(model.d), cov=cov)

def generate_input(n_fonts=5):
    fonts = [generate_font() for f in xrange(n_fonts)]
    for f in xrange(n_fonts):
        a, b = fonts[f], fonts[(f+1)%n_fonts]
        for p in numpy.linspace(0, 1, 10):
            print f, p
            batch_is = numpy.zeros((model.k, model.d), dtype=theano.config.floatX)
            batch_js = numpy.zeros((model.k,), dtype=numpy.int32)
            for z in xrange(model.k):
                batch_is[z] = a * (1-p) + b * p
                batch_js[z] = z

            yield batch_is, batch_js

print 'generating...'
frame = 0
for input_i, input_j in generate_input():
    pred = run_fn(input_i, input_j).reshape((input_i.shape[0], 64, 64))

    img = PIL.Image.new('L', (8 * 64, 8 * 64), 255)
    for z in xrange(model.k):
        x, y = z % 8, z // 8
        img_char = PIL.Image.fromarray(numpy.uint8(((1.0 - pred[z]) * 255)))
        img.paste(img_char, (x * 64, y * 64))

    img.save('font_%06d.png' % frame)
    frame += 1


