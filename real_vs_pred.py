import random
import numpy
import PIL, PIL.Image
import random
import model

data = model.get_data()
n, k = data.shape[0], data.shape[1]

model = model.Model(n, k)
model.try_load()
run_fn = model.get_run_fn()

train_set, test_set = model.sets()
chars = {}
for i, j in test_set:
    chars.setdefault(j, []).append(i)

batch_is = numpy.zeros((k,), dtype=numpy.int32)
batch_js = numpy.zeros((k,), dtype=numpy.int32)
for z in xrange(k):
    batch_is[z] = random.choice(chars[z]) # random.randint(0, n-1)
    batch_js[z] = z

batch_pred = run_fn(batch_is, batch_js).reshape((k, 64, 64))

img = PIL.Image.new('L', (12 * 64, 11 * 64), 255)
for z in xrange(k):
    x, y = (z*2) % 12, (z*2) // 12
    real = 255 - data[batch_is[z]][z]
    pred = numpy.uint8((1.0 - batch_pred[z])*255)
    img_char = PIL.Image.fromarray(real)
    img.paste(img_char, ((x + 0) * 64, y * 64))
    img_char = PIL.Image.fromarray(pred)
    img.paste(img_char, ((x + 1) * 64, y * 64))

img.save('real_vs_pred.png')


