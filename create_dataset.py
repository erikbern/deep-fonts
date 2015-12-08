import h5py
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw, PIL.ImageChops, PIL.ImageOps
import os
import random
import string
import numpy

def read_font(fn):
    font = PIL.ImageFont.truetype(fn, int(min(w0, h0) * 0.75))

    data = []
    for char in chars:
        print '...', char
        # Draw character
        img = PIL.Image.new("L", (w0, h0), 255)
        draw = PIL.ImageDraw.Draw(img)
        draw.text((0, 0), char, font=font)

        # Crop whitespace
        diff = PIL.ImageChops.difference(img, blank)
        img = img.crop(diff.getbbox())

        # Expand to square
        wi, hi, m = img.size[0], img.size[1], max(img.size)
        img_new = PIL.Image.new('L', (m, m), 255)
        img_new.paste(img, ((m-wi)/2, (m-hi)/2))

        # Resize to smaller
        img = img_new.resize((w, h), PIL.Image.ANTIALIAS)

        # Convert to numpy array
        matrix = numpy.array(img.getdata()).reshape((h, w))
        matrix = 255 - matrix
        data.append(matrix)

    return numpy.array(data)


def get_ttfs(d='scraper/fonts'):
    for dirpath, dirname, filenames in os.walk(d):
        for filename in filenames:
            if filename.endswith('.ttf'):
                yield os.path.join(dirpath, filename)

                
chars = string.uppercase + string.lowercase + string.digits

w, h = 64, 64
w0, h0 = 512, 512
blank = PIL.Image.new('L', (w0, h0), 255)

f = h5py.File('fonts.hdf5', 'w')
dset = f.create_dataset('fonts', (1, len(chars), h, w), chunks=(1, len(chars), h, w), maxshape=(None, len(chars), h, w), dtype='i1')

i = 0
for fn in get_ttfs():
    print fn
    try:
        data = read_font(fn)
    except IOError:
        print 'was not able to read', fn
        continue

    print data.shape
    dset.resize((i+1, len(chars), h, w))
    dset[i] = data
    i += 1
    f.flush()

f.close()
