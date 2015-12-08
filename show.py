import h5py, random
from matplotlib import pyplot

f = h5py.File('fonts.hdf5', 'r')
data = f['fonts']
print data.shape

for i in xrange(10):
    i = random.randint(0, data.shape[0]-1)
    j = random.randint(0, data.shape[1]-1)
    m = data[i][j]
    pyplot.matshow(m, cmap='gray')
    pyplot.show()


