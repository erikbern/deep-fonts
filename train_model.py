import random
import numpy
import theano
import model
import sys
import functools
import time

def iterate_minibatches(dataset, batch_size=2048):
    random.shuffle(dataset)
    for offset in xrange(0, len(dataset) - batch_size, batch_size):
        batch_fonts = numpy.zeros((batch_size,), dtype=numpy.int32)
        batch_chars = numpy.zeros((batch_size,), dtype=numpy.int32)
        batch_ds = numpy.zeros((batch_size, wh), dtype=theano.config.floatX)
        for z in xrange(batch_size):
            i, j = dataset[offset + z]
            batch_fonts[z] = i
            batch_chars[z] = j
            batch_ds[z] = data[i][j].flatten() * 1. / 255

        yield 1.0 * offset / len(dataset), batch_fonts, batch_chars, batch_ds

        
def iterate_run(dataset, fn, tag):
    total_loss, total_reg, total_count = 0, 0, 0
    for progress, input_font, input_char, output in iterate_minibatches(dataset):
        t0 = time.time()
        loss, reg = fn(input_font, input_char, output)
        t = time.time() - t0
        total_loss += float(loss)
        total_reg += float(reg)
        total_count += 1
        sys.stdout.write('%s: %6.2f%%, perf: %.6f + %.6f (last minibatch: %.6f + %.6f, %.3fs)\r' % (tag, 100.0 * progress, total_loss / total_count, total_reg / total_count, float(loss), float(reg), t))
        sys.stdout.flush()

    sys.stdout.write('\n')
    return total_loss / total_count

data = model.get_data()
n, k = data.shape[0], data.shape[1]
wh = data.shape[2] * data.shape[3]
model = model.Model(n, k, wh)
model.try_load()
train_fn_w_learning_rate = model.get_train_fn()
test_fn = model.get_test_fn()
run_fn = model.get_run_fn()
train_set, test_set = model.sets()

print 'training...'
for learning_rate in [1.0, 0.3, 0.1, 0.03, 0.01]:
    epoch = 0
    best_epoch = 0
    best_loss = float('inf')
    while True:
        print 'epoch', epoch, 'learning rate', learning_rate
        train_fn = functools.partial(train_fn_w_learning_rate, learning_rate)
        iterate_run(train_set, train_fn, 'train')
        loss = iterate_run(test_set, test_fn, 'test ')
        if loss < best_loss:
            print 'best loss'
            model.save()
            best_loss, best_epoch = loss, epoch
        if epoch - best_epoch > 2:
            break
        epoch += 1
