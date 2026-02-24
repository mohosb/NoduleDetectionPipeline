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

        if ct_data is not None:
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


class ToDeviceTransform(PipelinePart):
    def __init__(self, device='cpu'):
        self._device = torch.device(device)

    def __call__(self, *data, **params):
        ct_data, seg_data = data
        if ct_data is not None:
            ct_data = ct_data.to(self._device)
        if seg_data is not None:
            seg_data = seg_data.to(self._device)
        return (ct_data, seg_data), params

