import h5py, random, numpy
import PIL, PIL.Image

f = h5py.File('fonts.hdf5', 'r')
data = f['fonts']
print data.shape

i = random.randint(0, data.shape[0]-1)
for z in xrange(10):
    j = random.randint(0, data.shape[1]-1)
    img = PIL.Image.fromarray(numpy.uint8(255 - data[i][j]))
    img.show()



