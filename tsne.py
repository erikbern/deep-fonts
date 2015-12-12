import h5py
from sklearn.manifold import TSNE
from model import Model
import numpy
import PIL, PIL.Image

f = h5py.File('fonts.hdf5', 'r')
data = f['fonts']
n, k = data.shape[0], data.shape[1]
wh = data.shape[2] * data.shape[3]

model = Model(n, k, wh)
model.try_load()

ifb = model.input_font_bottleneck
X = numpy.maximum(ifb.W.get_value() + ifb.b.get_value(), 0)

tsne = TSNE(metric='cosine', verbose=99)
x = tsne.fit_transform(X[:5000])
x -= numpy.min(x, axis=0)
M = 10000
x *= M / numpy.max(x)

canvas = PIL.Image.new('L', (M + 64, M + 64), 255)
for i in xrange(len(x)):
    img = PIL.Image.fromarray(numpy.uint8(255 - data[i][0]))
    canvas.paste(img, (int(x[i][0]), int(x[i][1])))
canvas.save('tsne.png')
