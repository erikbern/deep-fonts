import h5py, random
from matplotlib import pyplot

f = h5py.File('fonts.hdf5', 'r')
data = f['fonts']
print data.shape

i = random.randint(0, data.shape[0]-1)
for z in xrange(10):
    j = random.randint(0, data.shape[1]-1)
    m = data[i][j]
    pyplot.matshow(m, cmap='gray')
    pyplot.show()


