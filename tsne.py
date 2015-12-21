import h5py
from sklearn.manifold import TSNE
from model import Model
import numpy
import PIL, PIL.Image, PIL.ImageOps

f = h5py.File('fonts.hdf5', 'r')
data = f['fonts']
n, k = data.shape[0], data.shape[1]
wh = data.shape[2] * data.shape[3]

model = Model(n, k, wh)
model.try_load()

X = model.get_font_embeddings()

print X

tsne = TSNE(verbose=99)
x = tsne.fit_transform(X[:5000])
x -= numpy.min(x, axis=0)
M = 10000
x *= M / numpy.max(x)

canvas = PIL.Image.new('L', (M + 128, M + 64), 255)
for i in xrange(len(x)):
    cx, cy = map(int, x[i])
    img_black = PIL.Image.new('L', (64, 64), 0)
    img = PIL.Image.fromarray(numpy.uint8(data[i][0]))
    canvas.paste(img_black, (cx, cy), img)
    #img = PIL.Image.fromarray(numpy.uint8(data[i][26]))
    #canvas.paste(img_black, (cx + 64, cy), img)
canvas.save('tsne.png')
