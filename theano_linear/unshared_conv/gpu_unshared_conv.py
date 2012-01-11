
class GpuFilterActs(FilterActs):
    """
    XXX

    """
    def c_support_code(self):
        cufile = open('filter_acts.cu')
        return cufile.read()

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, nodename, inputs, outputs, sub):
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
                "filters ndim (%%i) must be 7",
                %(filters)s->nd);
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

    def grad(self, inputs, goutputs):
        images, filters = inputs
        frows = filters.shape[3]
        fcols = filters.shape[4]
        gimages = None
        gfilters = GpuWeightActs(
                module_stride=self.module_stride,
                partial_sum=1)(images, goutputs[0], frows, fcols)
        return [gimages, gfilters]


class GpuWeightActs(theano.Op):
    """
    XXX
    """
    def __init__(self, module_stride, partial_sum):
        self.module_stride = module_stride
        self.partial_sum = partial_sum

    def _attributes(self):
        return (
                self.module_stride,
                self.partial_sum,
                )

    def __eq__(self, other):
        return (type(self) == type(other)
                and self._attributes() == other._attributes())

    def __hash__(self):
        return hash((type(self), self._attributes()))

    def __str__(self):
        return '%s{module_stride=%i,partial_sum=%i}' % (
                self.__class__.__name__,
                self.module_stride,
                self.partial_sum,
                )

    def make_node(self, images, hidacts, frows, fcols):
        if self.partial_sum != 1:
            # this corresponds to grad when doing convolution
            raise NotImplementedError('partial sum')
        frows = theano.tensor.as_tensor_variable(frows)
        fcols = theano.tensor.as_tensor_variable(fcols)
        if frows.dtype[:3] not in ('int', 'uin'):
            raise TypeError(frows)
        if fcols.dtype[:3] not in ('int', 'uin'):
            raise TypeError(frows)
        if frows.ndim:
            raise TypeError('frows should be scalar', frows)
        if fcols.ndim:
            raise TypeError('fcols should be scalar', fcols)

        igroups, icolors, irows, icols, icount = images.type.broadcastable
        hgroups, hcolors, hrows, hcols, hcount = hidacts.type.broadcastable
        otype = theano.sandbox.cuda.CudaNdarrayType(
                broadcastable=(hrows, hcols, icolors,
                    False, False, hgroups, hcolors))
        return theano.Apply(self,
                [images, hidacts, frows, fcols],
                [otype()])

    def c_support_code(self):
        cufile = open('weight_acts.cu')
        return cufile.read()

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, nodename, inames, onames, sub):
        images, hidacts, frows, fcols = inames
        dweights, = onames
        fail = sub['fail']
        moduleStride = str(self.module_stride)

        sio = StringIO.StringIO()

        print >> sio, """

        if (!CudaNdarray_is_c_contiguous(%(images)s))
        {
            //XXX: Alex's code actually supports the rightmost images
            //     dimension strided
            PyErr_Format(PyExc_NotImplementedError,
                "images not c contiguous");
            %(fail)s;
        }

        if (!CudaNdarray_is_c_contiguous(%(hidacts)s))
        {
            PyErr_Format(PyExc_NotImplementedError,
                "hidacts not c contiguous");
            %(fail)s;
        }

        if (%(images)s->nd != 5)
        {
            PyErr_Format(PyExc_TypeError,
                "images ndim (%%i) must be 5",
                %(images)s->nd);
            %(fail)s;
        }

        if (%(hidacts)s->nd != 5)
        {
            PyErr_Format(PyExc_TypeError,
                "hidacts ndim (%%i) must be 5",
                %(images)s->nd);
            %(fail)s;
        }

        if (%(frows)s->nd != 0)
        {
            PyErr_Format(PyExc_TypeError,
                "frows ndim (%%i) must be 0",
                %(frows)s->nd);
            %(fail)s;
        }

        if (%(fcols)s->nd != 0)
        {
            PyErr_Format(PyExc_TypeError,
                "fcols ndim (%%i) must be 0",
                %(fcols)s->nd);
            %(fail)s;
        }

        { // new scope, new vars

            int igroups           = CudaNdarray_HOST_DIMS(%(images)s)[0];
            int icolors_per_group = CudaNdarray_HOST_DIMS(%(images)s)[1];
            int irows             = CudaNdarray_HOST_DIMS(%(images)s)[2];
            int icols             = CudaNdarray_HOST_DIMS(%(images)s)[3];
            int icount            = CudaNdarray_HOST_DIMS(%(images)s)[4];

            int hgroups           = CudaNdarray_HOST_DIMS(%(hidacts)s)[0];
            int hcolors_per_group = CudaNdarray_HOST_DIMS(%(hidacts)s)[1];
            int hrows             = CudaNdarray_HOST_DIMS(%(hidacts)s)[2];
            int hcols             = CudaNdarray_HOST_DIMS(%(hidacts)s)[3];
            int hcount            = CudaNdarray_HOST_DIMS(%(hidacts)s)[4];

            int fmodulesR = hrows;
            int fmodulesC = hcols;
            int fcolors = icolors_per_group;
            int frows = ((dtype_%(frows)s *) (%(frows)s->data))[0];
            int fcols = ((dtype_%(fcols)s *) (%(fcols)s->data))[0];
            int fgroups = hgroups;
            int filters_per_group = hcolors_per_group;

            // XXX: use this parameter properly
            int paddingStart = 0;
            int imgStride = icount;
            float scaleTargets = 0.0;
            float scaleOutput = 1.0;
            int moduleStride = %(moduleStride)s;
            int partialSum = 1; // set to 0 for convolution.

            if (igroups != hgroups)
            {
                PyErr_Format(PyExc_ValueError,
                    "igroups != hgroups (%%i != %%i)",
                    igroups, hgroups);
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

            if (icount != hcount)
            {
                PyErr_Format(PyExc_ValueError,
                    "icount != hcount (%%i != %%i)",
                    icount,
                    hcount);
                %(fail)s;
            }

            if (!%(dweights)s)
            {
                Py_XDECREF(%(dweights)s);
                int dims[7];
                dims[0] = fmodulesR;
                dims[1] = fmodulesC;
                dims[2] = fcolors;
                dims[3] = frows;
                dims[4] = fcols;
                dims[5] = fgroups;
                dims[6] = filters_per_group;

                %(dweights)s = (CudaNdarray*)CudaNdarray_NewDims(7, dims);
                if (!%(dweights)s)
                {
                    %(fail)s;
                }
            }

            assert(CudaNdarray_is_c_contiguous(%(dweights)s));

            if (_weightActs(
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
                    CudaNdarray_DEV_DATA(%(hidacts)s),
                    CudaNdarray_DEV_DATA(%(dweights)s),
                    paddingStart,
                    moduleStride,
                    imgStride,
                    scaleTargets,
                    scaleOutput,
                    partialSum))
            {
                %(fail)s;
            }
        } // end bogus scope used for vars

        """

        return sio.getvalue() % locals()

