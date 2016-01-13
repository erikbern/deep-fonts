import numpy
import model
avg = numpy.mean(model.get_data()[:], axis=0)
model.draw_grid(avg / 255.).save('avg.png')
