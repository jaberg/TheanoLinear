import numpy
import theano
import StringIO

from ..linear import LinearTransform


class ImageLocalDot(LinearTransform):
    """
    ImageLocalDot is an linear operation computationlly similar to
    convolution in the spatial domain, except that whereas convolution
    applying a single filter or set of filters across an image, the
    ImageLocalDot has different filterbanks for different points in the image.

    Mathematically, this is a general linear transform except for a
    restriction that filters are 0 outside of a spatially localized patch
    within the image.

    """

    def __init__(self, filters, img_shape,
            subsample=(1, 1),
            border_mode='valid',
            filters_shape=None,
            message=""):
        super(LConv, self).__init__([filters])
        self._filters = filters
        if filters_shape is None:
            self._filters_shape = tuple(filters.get_value().shape)
        else:
            self._filters_shape = tuple(filters_shape)
        self._img_shape = tuple(img_shape)
        self._subsample = tuple(subsample)
        self._border_mode = border_mode
        if message:
            self._message = message
        else:
            self._message = filters.name
        if not len(self._img_shape)==4:
            raise TypeError('need 4-tuple shape', self._img_shape)
        if not len(self._filters_shape)==4:
            raise TypeError('need 4-tuple shape', self._filters_shape)

    def _lmul(self, x, T):
        if T:
            # left-multiply transpose of self with x
            raise NotImplementedError()
        else:
            # left-multiply self with x
            raise NotImplementedError()

    def _row_shape(self):
        return self._img_shape[1:]

    def _col_shape(self):
        rows_cols = ConvOp.getOutputShape(
                self._img_shape[2:],
                self._filters_shape[2:],
                self._subsample,
                self._border_mode)
        rval = (self._filters_shape[0],)+tuple(rows_cols)
        return rval

    def _tile_columns(self, scale_each=True, **kwargs):
        return pylearn.io.image_tiling.tile_slices_to_image(
                self._filters.get_value()[:,:,::-1,::-1].transpose(0,2,3,1),
                scale_each=scale_each,
                **kwargs)

    def print_status(self):
        print ndarray_status(
                self._filters.get_value(borrow=True),
                msg='LConv{%s}'%self._message)


class FilterActs(theano.Op):
    """
    Images of shape: colors x 
    Filters are of shape:
        channels
    """
    def __init__(self,
            module_stride=1,
            ):
        self.module_stride = module_stride

    def _attributes(self):
        return (
                self.module_stride,
                )

    def __eq__(self, other):
        return (type(self) == type(other)
                and self._attributes() == other._attributes())

    def __hash__(self):
        return hash((type(self), self._attributes()))

    def __str__(self):
        return '%s{module_stride=%i}' % (
                self.__class__.__name__,
                self.module_stride,
                )

    def make_node(self, images, filters):
        return theano.gof.Apply(self,
                [images, filters],
                [images.type()])

    def perform(self, node, iargs, ostor):
        images, filters = iargs

        igroups, icolors_per_group, irows, icols, icount = images.shape
        fmodulesR, fmodulesC, fcolors, frows, fcols = filters.shape[:-2]
        fgroups, filters_per_group = filters.shape[-2:]

        if irows != icols:
            raise NotImplementedError("non-square image argument",
                    (irows, icols))
        if frows != fcols:
            raise NotImplementedError("non-square filter shape",
                    (frows, fcols))
        if fmodulesR != fmodulesC:
            raise NotImplementedError('non-square filter grouping',
                    (fmodulesR, fmodulesC))
        if icolors_per_group != fcolors:
            raise ValueError("color counts don't match",
                    (icolors_per_group, fcolors))

        target = numpy.zeros(
                (fgroups, filters_per_group, fmodulesR, fmodulesC, icount),
                dtype=images.dtype)

        for mR in xrange(fmodulesR):
            for mC in xrange(fmodulesC):
                for gg in xrange(igroups):
                    img_r_offset = mR * self.module_stride
                    img_c_offset = mC * self.module_stride
                    rc_images = images[gg, :,
                            img_r_offset:img_r_offset + frows,
                            img_c_offset:img_c_offset + fcols,
                            :]
                    rc_filters = filters[mR, mC, :, :, :, gg, :]
                    # rc_images are fcolors x frows x fcols x count
                    # rc_filters are fcolors x frows x fcols x fpg
                    rc_target = numpy.dot(
                        rc_filters.reshape(-1, filters_per_group).T,
                        rc_images.reshape(-1, icount))
                    target[gg, :, mR, mC, :] = rc_target
        ostor[0][0] = target


class GpuFilterActs(FilterActs):
    """

    """
    def c_support_code(self):
        cufile = open('filter_acts.cu')
        return cufile.read()

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, name, inputs, outputs, sub):
        #z_out = alpha * dot(x,y) + beta * z_in
        #inplace version, set set z_out = z_in
        #not inplace version, we copy z_in to z_out.
        images, filters, = inputs
        responses, = outputs
        fail = sub['fail']
        moduleStride = str(self.module_stride)
        sio = StringIO.StringIO()

        print >> sio, """

        //XXX: actually the rightmost images dimension can be strided
        if (!CudaNdarray_is_c_contiguous(%(images)s))
        {
            PyErr_Format(PyExc_NotImplementedError,
                "images not c contiguous");
            %(fail)s;
        }

        if (!CudaNdarray_is_c_contiguous(%(filters)s))
        {
            PyErr_Format(PyExc_NotImplementedError,
                "filters not c contiguous");
            %(fail)s;
        }

        if (%(images)s->nd != 5)
        {
            PyErr_Format(PyExc_TypeError,
                "images ndim (%%i) must be 5",
                %(images)s->nd);
            %(fail)s;
        }

        if (%(filters)s->nd != 7)
        {
            PyErr_Format(PyExc_TypeError,
                "images ndim (%%i) must be 5",
                %(images)s->nd);
            %(fail)s;
        }

        { // new scope, new vars

            int igroups           = CudaNdarray_HOST_DIMS(%(images)s)[0];
            int icolors_per_group = CudaNdarray_HOST_DIMS(%(images)s)[1];
            int irows             = CudaNdarray_HOST_DIMS(%(images)s)[2];
            int icols             = CudaNdarray_HOST_DIMS(%(images)s)[3];
            int icount            = CudaNdarray_HOST_DIMS(%(images)s)[4];

            int fmodulesR         = CudaNdarray_HOST_DIMS(%(filters)s)[0];
            int fmodulesC         = CudaNdarray_HOST_DIMS(%(filters)s)[1];
            int fcolors           = CudaNdarray_HOST_DIMS(%(filters)s)[2];
            int frows             = CudaNdarray_HOST_DIMS(%(filters)s)[3];
            int fcols             = CudaNdarray_HOST_DIMS(%(filters)s)[4];
            int fgroups           = CudaNdarray_HOST_DIMS(%(filters)s)[5];
            int filters_per_group = CudaNdarray_HOST_DIMS(%(filters)s)[6];

            // XXX: use this parameter properly
            int paddingStart = 0;
            int imgStride = icount;
            float scaleTargets = 0.0;
            float scaleOutput = 1.0;
            bool conv = false;

            if (igroups != fgroups)
            {
                PyErr_Format(PyExc_ValueError,
                    "igroups != fgroups (%%i != %%i)",
                    igroups, fgroups);
                %(fail)s;
            }

            if (icolors_per_group != fcolors)
            {
                PyErr_Format(PyExc_ValueError,
                    "icolors_per_group != fcolors (%%i != %%i)",
                    icolors_per_group,
                    fcolors);
                %(fail)s;
            }

            if (!%(responses)s)
            {
                Py_XDECREF(%(responses)s);
                int dims[5];
                dims[0] = fgroups;
                dims[1] = filters_per_group;
                dims[2] = fmodulesR;
                dims[3] = fmodulesC;
                dims[4] = icount;
                %(responses)s = (CudaNdarray*)CudaNdarray_NewDims(5, dims);
                if (!%(responses)s)
                {
                    %(fail)s;
                }
            }

            assert(CudaNdarray_is_c_contiguous(%(responses)s));

            if (_filterActs(
                    igroups,
                    icolors_per_group,
                    irows,
                    icols,
                    icount,
                    fmodulesR,
                    fmodulesC,
                    frows,
                    fcols,
                    filters_per_group,
                    CudaNdarray_DEV_DATA(%(images)s),
                    CudaNdarray_DEV_DATA(%(filters)s),
                    CudaNdarray_DEV_DATA(%(responses)s),
                    paddingStart,
                    %(moduleStride)s,
                    imgStride,
                    scaleTargets,
                    scaleOutput,
                    conv))
            {
                %(fail)s;
            }
        } // end bogus scope used for vars

        """

        return sio.getvalue() % locals()

    def perform(self, *args, **kwargs):
        return theano.Op.perform(self, *args, **kwargs)
