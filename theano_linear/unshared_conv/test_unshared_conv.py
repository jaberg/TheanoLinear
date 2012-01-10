
import numpy
import theano
import unittest

from .unshared_conv import FilterActs

class TestFilterActs(unittest.TestCase):
    ishape = (1, 4, 4, 2) # 2 4x4 greyscale images
    fshape = (2, 2, 1, 3, 3, 5) # 5 3x3 filters at each location in a 2x2 grid
    module_stride = 1

    def setUp(self):
        self.fa = FilterActs(self.module_stride)
        self.s_images = theano.shared(numpy.random.rand(*self.ishape))
        self.s_filters = theano.shared(numpy.random.rand(*self.fshape))

    def test_type(self):
        out = self.fa(self.s_images, self.s_filters)
        assert out.type == self.s_images.type

        f = theano.function([], out)
        outval = f()
        assert len(outval.shape) == len(self.ishape)
        assert outval.dtype == self.s_images.get_value(borrow=True).dtype

    def test_linearity(self):
        out = self.fa(self.s_images, self.s_filters)

        t = theano.tensor.scalar()
        out2 = self.fa(self.s_images * t, self.s_filters)

        images3 = theano.shared(numpy.random.rand(*self.ishape))
        out3 = self.fa(images3, self.s_filters) + out

        out4 = self.fa(images3 + self.s_images, self.s_filters)

        f = theano.function([t], [out * t, out2, out3, out4])
        outval, out2val, out3val, out4val = f(3.6)
        assert numpy.allclose(outval, out2val)
        assert numpy.allclose(out3val, out4val)

