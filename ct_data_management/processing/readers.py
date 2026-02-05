import torch
import monai.transforms as mt
import numpy as np
from .pipeline import PipelinePart


class DICOMDataAnomalyError(Exception):
    pass


class NSCLCRadiomicsReader(PipelinePart):
    _CT_BACKEND = mt.LoadImage('ITKReader', image_only=True, ensure_channel_first=True, dtype=np.float32)
    _SEG_BACKEND = mt.LoadImage('PydicomReader', image_only=True, ensure_channel_first=True, dtype=np.float32)

    def __call__(self, *data, **params):
        ct_path, seg_path = data

        ct_data = None
        seg_data = None

        if ct_path is not None:
            ct_data = self._CT_BACKEND(ct_path)

        if seg_path is not None:
            seg_data = self._SEG_BACKEND(seg_path)

            lung_segments = []
            nodule_segments = []
            for seg_label, seg_number in seg_data.meta['labels'].items():
                seg_label = seg_label.strip().lower()
                if 'lung' in seg_label:
                    lung_segments.append(seg_number)
                elif seg_label == 'gtv-1':
                    nodule_segments.append(seg_number)

            if len(lung_segments) == 0:
                raise DICOMDataAnomalyError('No Lung instance could be detected in SEG DICOM file.')
    
            # Union Region of Interest instances into a single instance and separate the nodule(s)
            roi_data = seg_data[lung_segments + nodule_segments].sum(0).clamp_(0, 1)
            nodule_data = seg_data[nodule_segments].sum(0).clamp_(0, 1)
            seg_data = torch.stack((roi_data, nodule_data))

        return (ct_data, seg_data), params


class NLSTReader(PipelinePart):
    pass


class LIDCIDRIReader(PipelinePart):
    pass


class PAXReader(PipelinePart):
    pass

