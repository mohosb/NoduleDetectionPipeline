import torch
import torch.nn.functional as F
import monai.transforms as mt
from .pipeline import PipelinePart


class DataAnomalyError(Exception):
    pass


class IDGenerator(PipelinePart):
    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_header = params.get('ct_header', None)
        if ct_header is not None:
            params['id'] = ct_header.SeriesInstanceUID
        return data, params


class FilterSegmentsTransform(PipelinePart):
    def __init__(self, target_labels=tuple(), min_num_segments=0):
        self._target_labels = target_labels
        self._min_num_segments = min_num_segments

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        seg_header_list = params.get('seg_header_list', None)
        seg_data_list   = data.get('seg_list')

        if seg_header_list is None:
            return data, params

        new_seg_data_list = []
        for seg_header, seg_data in zip(seg_header_list, seg_data_list):
            target_segments = set()
            if 'SegmentSequence' in seg_header:
                for item in seg_header.SegmentSequence:
                    seg_num = item.SegmentNumber - 1  # DICOM files start indexing from 1

                    if 'SegmentLabel' in item:
                        seg_label = item.SegmentLabel
                    elif 'SegmentDescription' in item:
                        seg_label = item.SegmentDescription
                    else:
                        seg_label = ''

                    seg_label = seg_label.lower()
                    for label_pattern in self._target_labels:
                        if label_pattern in seg_label:
                            target_segments.add(seg_num)

            # Drop only out-of-range indices rather than the entire SEG file.
            target_segments = {i for i in target_segments if i < seg_data.size(0)}

            if len(target_segments) < self._min_num_segments:
                continue

            seg_data = seg_data[list(target_segments)].sum(0, keepdim=True).clamp_(0, 1)
            new_seg_data_list.append(seg_data)

        if len(new_seg_data_list) == 0:
            raise DataAnomalyError('No correct segmentation could be found.')

        data['seg_list'] = new_seg_data_list
        return data, params


class ClipAndNormTransform(PipelinePart):
    def __init__(self, clip_min, clip_max):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_data = data.get('ct')

        if ct_data is not None:
            ct_data.clip_(self.clip_min, self.clip_max)

            ct_data_min = ct_data.min()
            ct_data_max = ct_data.max()
            denom = ct_data_max - ct_data_min
            if denom == 0:
                ct_data = torch.zeros_like(ct_data)
            else:
                ct_data = (ct_data - ct_data_min) / denom

            data['ct'] = ct_data

        return data, params


class OrientTransform(PipelinePart):
    def __init__(self, orientation='RAS'):
        self._backend = mt.Orientation(axcodes=orientation, labels=None)

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        if data.get('ct') is not None:
            data['ct'] = self._backend(data['ct'])
        if data.get('seg_list') is not None:
            data['seg_list'] = [self._backend(s) for s in data['seg_list']]
        return data, params


class ResampleTransform(PipelinePart):
    def __init__(self, spacing=(1., 1., 1.), ct_mode='bilinear', seg_mode='nearest'):
        self._ct_backend  = mt.Spacing(pixdim=spacing, mode=ct_mode)
        self._seg_backend = mt.ResampleToMatch(mode=seg_mode, padding_mode='zeros')

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        if data.get('ct') is not None:
            data['ct'] = self._ct_backend(data['ct'])
        if data.get('seg_list') is not None:
            data['seg_list'] = [self._seg_backend(s, data['ct']) for s in data['seg_list']]
        return data, params


class MergeSegmentsTransform(PipelinePart):
    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        seg_data_list = data.pop('seg_list', None)

        if not seg_data_list:
            data['seg'] = None
            return data, params

        # If there is only one SEG (like in NSCLC), just use it directly
        if len(seg_data_list) == 1:
            data['seg'] = seg_data_list[0]
        else:
            data['seg'] = torch.stack(seg_data_list, dim=0).sum(dim=0).clamp_(0, 1)

        return data, params


class ToDeviceTransform(PipelinePart):
    def __init__(self, device='cpu'):
        self._device = torch.device(device)

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        if data.get('ct') is not None:
            data['ct'] = data['ct'].to(self._device)
        if data.get('seg_list') is not None:
            data['seg_list'] = [s.to(self._device) for s in data['seg_list']]
        if data.get('seg') is not None:
            data['seg'] = data['seg'].to(self._device)
        return data, params
