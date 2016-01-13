import random
import numpy
import model

data = model.get_data()
n = data.shape[0]
alphabet = numpy.zeros((62, 64, 64))
for z in xrange(62):
    alphabet[z] = random.choice(data)[z] * 1.0/255

model.draw_grid(alphabet).save('alphabet.png')
