import h5py
import PIL, PIL.ImageFont, PIL.Image, PIL.ImageDraw, PIL.ImageChops, PIL.ImageOps
import os
import random
import string
import numpy
import sys

w, h = 64, 64
w0, h0 = 256, 256

chars = string.uppercase + string.lowercase + string.digits

blank = PIL.Image.new('L', (w0*5, h0*3), 255)

def read_font(fn):
    font = PIL.ImageFont.truetype(fn, min(w0, h0))

    # We need to make sure we scale down the fonts but preserve the vertical alignment
    min_ly = float('inf')
    max_hy = float('-inf')
    max_width = 0
    imgs = []

    for char in chars:
        print '...', char
        # Draw character
        img = PIL.Image.new("L", (w0*5, h0*3), 255)
        draw = PIL.ImageDraw.Draw(img)
        draw.text((w0, h0), char, font=font)

        # Get bounding box
        diff = PIL.ImageChops.difference(img, blank)
        lx, ly, hx, hy = diff.getbbox()
        min_ly = min(min_ly, ly)
        max_hy = max(max_hy, hy)
        max_width = max(max_width, hx - lx)
        imgs.append((lx, hx, img))

    print 'crop dims:', max_hy - min_ly, max_width
    scale_factor = min(1.0 * h / (max_hy - min_ly), 1.0 * w / max_width)
    data = []

    for lx, hx, img in imgs:
        img = img.crop((lx, min_ly, hx, max_hy))

        # Resize to smaller
        new_width = (hx-lx) * scale_factor
        new_height = (max_hy - min_ly) * scale_factor
        img = img.resize((int(new_width), int(new_height)), PIL.Image.ANTIALIAS)

        # Expand to square
        img_sq = PIL.Image.new('L', (w, h), 255)
        offset_x = (w - new_width)/2
        offset_y = (h - new_height)/2
        print offset_x, offset_y
        img_sq.paste(img, (int(offset_x), int(offset_y)))

        # Convert to numpy array
        matrix = numpy.array(img_sq.getdata()).reshape((h, w))
        matrix = 255 - matrix
        data.append(matrix)

    return numpy.array(data)


def get_ttfs(d='scraper/fonts'):
    for dirpath, dirname, filenames in os.walk(d):
        for filename in filenames:
            if filename.endswith('.ttf') or filename.endswith('.otf'):
                yield os.path.join(dirpath, filename)

                
f = h5py.File('fonts.hdf5', 'w')
dset = f.create_dataset('fonts', (1, len(chars), h, w), chunks=(1, len(chars), h, w), maxshape=(None, len(chars), h, w), dtype='u1')

i = 0
for fn in get_ttfs(d=sys.argv[1]):
    print fn
    try:
        data = read_font(fn)
    except: # IOError:
        print 'was not able to read', fn
        continue

    print data.shape
    dset.resize((i+1, len(chars), h, w))
    dset[i] = data
    i += 1
    f.flush()

f.close()
