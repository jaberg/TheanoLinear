
import unittest

import numpy

import theano

from localdot import LocalDot


class TestLocalDot(unittest.TestCase):
    def setUp(self):
        # XXX: use theano random
        numpy.random.rand.seed(234)

    def rand(self, shp):
        return numpy.random.rand(shp).astype('float32')

    def test_32x32(self):
        channels = 3
        bsize = 10     # batch size
        imshp = (32, 32)
        ksize = 5
        nkern_per_group = 16
        subsample_strides = 1, 2, 3, 4
        ngroups = 1, 3
        icount = 1

        for subsample_stride in subsample_strides:
            for ngroup in ngroups:
                fModulesR = (imshp[0] - ksize + 1) // subsample_stride
                fModulesC = fModulesR
                fshape = (fModulesR, fModulesC, channels // ngroup,
                        ksize, ksize, ngroups, nkern_per_group)
                ishape = (ngroup, channels // ngroup, imshp[0], imshp[1],
                        icount)
                filters = self.rand(fshape)
                LocalDot(filters, ishape,
                        subsample=(subsample_strides, subsample_strides))

