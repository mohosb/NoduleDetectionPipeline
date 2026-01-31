
import torch
import torch.nn.functional as F
import monai.transforms as mt
from .pipeline import PipelinePart


class ClipAndNormTransform(PipelinePart):
    def __init__(self, clip_min, clip_max):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, *data, **params):
        ct_data, seg_data = data

        ct_data.clip_(self.clip_min, self.clip_max)

        ct_data_min = ct_data.min()
        ct_data_max = ct_data.max()
        ct_data = (ct_data - ct_data_min) / (ct_data_max - ct_data_min)

        return (ct_data, seg_data), params


class OrientTransform(PipelinePart):
    def __init__(self, orientation='RAS'):
        self._backend = mt.Orientation(axcodes=orientation, labels=None)

    def __call__(self, *data, **params):
        ct_data, seg_data = data
        if ct_data is not None:
            ct_data = self._backend(ct_data)
        if seg_data is not None:
            seg_data = self._backend(seg_data)
        return (ct_data, seg_data), params


class ResampleTransform(PipelinePart):
    def __init__(self, spacing=(1., 1., 1.), ct_mode='bilinear', seg_mode='nearest'):
        self._ct_backend = mt.Spacing(pixdim=spacing, mode=ct_mode)
        self._seg_backend = mt.ResampleToMatch(mode=seg_mode)

    def __call__(self, *data, **params):
        ct_data, seg_data = data
        if ct_data is not None:
            ct_data = self._ct_backend(ct_data)
        if seg_data is not None:
            seg_data = self._seg_backend(seg_data, ct_data)
        return (ct_data, seg_data), params


class AutoCropTransform(PipelinePart):
    def __init__(self, scale_factor=10, threshold=0.5, padding=0):
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.padding = padding

    def __call__(self, *data, **params):
        ct_data, seg_data = data

        if ct_data is not None:
            ct_activity = F.avg_pool3d(ct_data[None, ...], kernel_size=self.scale_factor, stride=self.scale_factor, ceil_mode=True)

            mask = ct_activity > self.threshold
            mask = mask[0, 0]

            if not mask.any():
                return (ct_data, seg_data), params

            x_idxs = torch.nonzero(mask.amax(dim=(1, 2)), as_tuple=True)[0]
            y_idxs = torch.nonzero(mask.amax(dim=(0, 2)), as_tuple=True)[0]
            z_idxs = torch.nonzero(mask.amax(dim=(0, 1)), as_tuple=True)[0]

            def get_slice(idxs, original_dim):
                start = max(0, idxs[0].item() * self.scale_factor - self.padding)
                end = min(original_dim, (idxs[-1].item() + 1) * self.scale_factor - 1)
                return slice(start, end)

            x_slice = get_slice(x_idxs, ct_data.size(-3))
            y_slice = get_slice(y_idxs, ct_data.size(-2))
            z_slice = get_slice(z_idxs, ct_data.size(-1))

            ct_data = ct_data[..., x_slice, y_slice, z_slice]

            if seg_data is not None:
                seg_data = seg_data[..., x_slice, y_slice, z_slice]


        #---------------------
        # Basic Lung Segmentation with Thresholding
        #lung_seg_data = (ct_data > 0.02857142857142857).logical_and(ct_data < 0.42857142857142855).float()
        #seg_data = torch.cat((seg_data, lung_seg_data), dim=0)
        #---------------------


        return (ct_data, seg_data), params
