
import numpy
import theano
import unittest

from .unshared_conv import FilterActs
from .unshared_conv import GpuFilterActs

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


class TestFilterActsF32(TestFilterActs):
    dtype = 'float32'

from theano.sandbox.cuda.var import float32_shared_constructor

class TestGpuFilterActs(unittest.TestCase):
    ishape = (1, 1, 4, 4, 2) # 2 4x4 greyscale images
    fshape = (2, 2, 1, 3, 3, 1, 16) # 5 3x3 filters at each location in a 2x2 grid
    module_stride = 1
    dtype = 'float32'

    def setUp(self):
        self.gfa = GpuFilterActs(self.module_stride)
        self.gpu_images = float32_shared_constructor(
                numpy.random.rand(*self.ishape).astype(self.dtype))
        self.gpu_filters = float32_shared_constructor(
                numpy.random.rand(*self.fshape).astype(self.dtype))

    def test_shape(self):
        gpuout = self.gfa(self.gpu_images, self.gpu_filters)
        f = theano.function([], gpuout)
        outval = f()
        #print 'RAN! output shape: ', outval.shape
        assert outval.shape == (
                self.fshape[-2], self.fshape[-1],
                self.fshape[0], self.fshape[1],
                self.ishape[-1])

class TestMatchFilterActs(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(77)

    def run_match(self, images, filters, module_stride, retvals=False):

        gfa = GpuFilterActs(module_stride)
        fa = FilterActs(module_stride)

        gpu_images = float32_shared_constructor(images)
        gpu_filters = float32_shared_constructor(filters)
        cpu_images = theano.shared(images)
        cpu_filters = theano.shared(filters)

        gpu_out = gfa(gpu_images, gpu_filters)
        cpu_out = fa(cpu_images, cpu_filters)

        f = theano.function([], [cpu_out, gpu_out])
        cpuval, gpuval = f()
        gpuval = numpy.asarray(gpuval)

        if retvals:
            return cpuval, gpuval
        else:
            #print 'run_match: cpu shape', cpuval.shape
            #print 'run_match: gpu shape', gpuval.shape
            assert cpuval.shape == gpuval.shape
            assert numpy.allclose(cpuval, gpuval)

    def run_match_shape(self, ishape, fshape, module_stride, dtype='float32'):
        return self.run_match(
            images=numpy.random.rand(*ishape).astype(dtype),
            filters=numpy.random.rand(*fshape).astype(dtype),
            module_stride=module_stride)

    def test_small_random(self):
        self.run_match_shape(
            ishape = (1, 1, 4, 4, 2),
            fshape = (2, 2, 1, 3, 3, 1, 16),
            module_stride = 1)

    def test_small_random_colors(self):
        self.run_match_shape(
            ishape = (1, 6, 4, 4, 2),
            fshape = (2, 2, 6, 3, 3, 1, 16),
            module_stride = 1)

    def test_small_random_groups(self):
        self.run_match_shape(
            ishape = (5, 6, 4, 4, 2),
            fshape = (2, 2, 6, 3, 3, 5, 16),
            module_stride = 1)

    def test_small_random_module_stride(self):
        self.run_match_shape(
            ishape = (4, 6, 5, 5, 1),
            fshape = (2, 2, 6, 3, 3, 4, 16),
            module_stride = 2)

    def test_med_random_module_stride(self):
        self.run_match_shape(
            ishape = (4, 6, 32, 32, 1),
            fshape = (12, 12, 6, 3, 3, 4, 16),
            module_stride = 2)


    def _blah_topcorner_filter1(self):
        ishape = (1, 1, 4, 4, 2)
        fshape = (2, 2, 1, 3, 3, 1, 16)
        images = numpy.random.rand(*ishape).astype('float32')
        filters = numpy.random.rand(*fshape).astype('float32')
        filters *= 0
        filters[0,0,0,0,0,0,0] = 1
        self.run_match(images, filters, 1)

    def _blah_botcorner_filter1(self):
        ishape = (1, 1, 4, 4, 2)
        fshape = (2, 2, 1, 3, 3, 1, 16)
        images = numpy.random.rand(*ishape).astype('float32')
        filters = numpy.random.rand(*fshape).astype('float32')
        filters *= 0
        filters[1,1,0,0,0,0,0] = 1
        cpuval, gpuval = self.run_match(images, filters, 1, retvals=True)
        print images
        print cpuval[:, :, 1, 1, :]
        print gpuval[:, :, 1, 1, :]
