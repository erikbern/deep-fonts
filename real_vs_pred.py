import random
import numpy
import random
import model

data = model.get_data()
n, k = data.shape[0], data.shape[1]

m = model.Model(n, k)
m.try_load()
run_fn = m.get_run_fn()

train_set, test_set = m.sets()
chars = {}
for i, j in test_set:
    chars.setdefault(j, []).append(i)

batch_is = numpy.zeros((k,), dtype=numpy.int32)
batch_js = numpy.zeros((k,), dtype=numpy.int32)
for z in xrange(k):
    batch_is[z] = random.choice(chars[z]) # random.randint(0, n-1)
    batch_js[z] = z

batch_pred = run_fn(batch_is, batch_js)
combined = numpy.zeros((2*k, 64 * 64))
for z in xrange(k):
    combined[2*z] = data[batch_is[z]][z].flatten() * 1.0 / 255
    combined[2*z+1] = batch_pred[z]

model.draw_grid(combined).save('real_vs_pred.png')
