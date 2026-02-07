
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


class CropLungRegionTransform(PipelinePart):
    def __init__(self, scale_factor=0.25, min_value=-960, max_value=-400, padding=10):
        self.downsample_factor = scale_factor
        self.min_value = min_value
        self.max_value = max_value
        self.padding = padding

    def __call__(self, *data, **params):
        ct_data, seg_data = data

        if ct_data is not None:
            print('Original Shape:')
            print(ct_data.shape)


            ct_coarse = F.interpolate(ct_data[None, ...], scale_factor=1/8, mode='trilinear')
            ct_coarse = ct_coarse.squeeze(0)

            mask = (ct_coarse > self.min_value) & (ct_coarse < self.max_value)

            mask = F.interpolate(mask[None, ...].float(), scale_factor=1/4, mode='trilinear')
            mask = mask.squeeze(0)
            mask = mask > 0.8

            #return (ct_coarse, None), params
            #return (mask, None), params

            if not mask.any():
                print('Warning: No Lung region was found during coarse segmentation')
                return (ct_data, seg_data), params

            x_idxs = torch.nonzero(mask.amax(dim=(1, 2)), as_tuple=True)[0]
            y_idxs = torch.nonzero(mask.amax(dim=(0, 2)), as_tuple=True)[0]
            z_idxs = torch.nonzero(mask.amax(dim=(0, 1)), as_tuple=True)[0]

            #upsample_factor = 1 / self.downsample_factor
            upsample_factor = 32

            def get_slice(idxs, original_dim):
                start = max(0, idxs[0].item() * upsample_factor - self.padding)
                end = min(original_dim, (idxs[-1].item() + 1) * upsample_factor - 1 + self.padding)
                return slice(int(start), int(end))

            x_slice = get_slice(x_idxs, ct_data.size(-3))
            y_slice = get_slice(y_idxs, ct_data.size(-2))
            z_slice = get_slice(z_idxs, ct_data.size(-1))

            ct_data = ct_data[..., x_slice, y_slice, z_slice]

            if seg_data is not None:
                seg_data = seg_data[..., x_slice, y_slice, z_slice]

        print('New Shape:')
        print(ct_data.shape)

        return (ct_data, seg_data), params
