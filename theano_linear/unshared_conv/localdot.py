"""
XXX
"""

from ..linear import LinearTransform
from unshared_conv import FilterActs, ImgActs, WeightActs

class LocalDot(LinearTransform):
    """
    LocalDot is an linear operation computationlly similar to
    convolution in the spatial domain, except that whereas convolution
    applying a single filter or set of filters across an image, the
    LocalDot has different filterbanks for different points in the image.

    Mathematically, this is a general linear transform except for a
    restriction that filters are 0 outside of a spatially localized patch
    within the image.

    Image shape is 5-tuple:
        color_groups
        colors_per_group
        rows
        cols
        images

    Filterbank shape is 7-tuple (!)
        row_positions
        col_positions
        colors_per_group
        height
        width
        filter_groups
        filters_per_group

    The result of left-multiplication a 5-

    """

    def __init__(self, filters, img_shape,
            subsample=(1, 1),
            border_mode='valid',
            padding_start=None,
            filters_shape=None,
            message=""):
        super(LConv, self).__init__([filters])
        self._filters = filters
        if filters_shape is None:
            self._filters_shape = tuple(filters.get_value(borrow=True).shape)
        else:
            self._filters_shape = tuple(filters_shape)
        self._img_shape = tuple(img_shape)
        self._subsample = tuple(subsample)
        self._border_mode = border_mode
        self._padding_start = padding_start
        if message:
            self._message = message
        else:
            self._message = filters.name
        if not len(self._img_shape) == 5:
            raise TypeError('need 5-tuple image shape', self._img_shape)
        if not len(self._filters_shape) == 7:
            raise TypeError('need 7-tuple filter shape', self._filters_shape)

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



