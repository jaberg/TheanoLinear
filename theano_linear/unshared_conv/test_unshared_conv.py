
import unittest

import numpy

import theano
from theano.sandbox.cuda.var import float32_shared_constructor
from theano.tests.unittest_tools import verify_grad

from .unshared_conv import FilterActs

class TestFilterActs(unittest.TestCase):
    ishape = (1, 1, 4, 4, 2) # 2 4x4 greyscale images
    fshape = (2, 2, 1, 3, 3, 1, 5) # 5 3x3 filters at each location in a 2x2 grid
    module_stride = 1
    dtype = 'float64'

    def setUp(self):
        self.fa = FilterActs(self.module_stride)
        self.s_images = theano.shared(
                numpy.random.rand(*self.ishape).astype(self.dtype))
        self.s_filters = theano.shared(
                numpy.random.rand(*self.fshape).astype(self.dtype))

    def test_type(self):
        out = self.fa(self.s_images, self.s_filters)
        assert out.type == self.s_images.type

        f = theano.function([], out)
        outval = f()
        assert len(outval.shape) == len(self.ishape)
        assert outval.dtype == self.s_images.get_value(borrow=True).dtype

    def test_linearity(self):
        out = self.fa(self.s_images, self.s_filters)

        t = theano.tensor.scalar(dtype=self.dtype)
        out2 = self.fa(self.s_images * t, self.s_filters)

        images3 = theano.shared(
                numpy.random.rand(*self.ishape).astype(self.dtype))
        out3 = self.fa(images3, self.s_filters) + out

        out4 = self.fa(images3 + self.s_images, self.s_filters)

        f = theano.function([t], [out * t, out2, out3, out4],
                allow_input_downcast=True)
        outval, out2val, out3val, out4val = f(3.6)
        assert numpy.allclose(outval, out2val)
        assert numpy.allclose(out3val, out4val)

    def test_shape(self):
        out = self.fa(self.s_images, self.s_filters)
        f = theano.function([], out)
        outval = f()
        assert outval.shape == (self.fshape[-2],
                self.fshape[-1],
                self.fshape[0], self.fshape[1],
                self.ishape[-1])

    def test_grad(self):
        try:
            verify_grad(self.fa,
                    [self.s_images.get_value(),
                        self.s_filters.get_value()])
        except verify_grad.E_grad, e:
            print e.num_grad.gf
            print e.analytic_grad
            raise


class TestFilterActsF32(TestFilterActs):
    dtype = 'float32'


