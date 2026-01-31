import pydicom
from pathlib import Path
import torch
import monai.transforms as mt
import numpy as np
from .pipeline import PipelinePart


class DICOMDataAnomalyError(Exception):
    pass


class DICOMFileSystemReader(PipelinePart):
    def find_series_paths(self, read_path):
        read_path = Path(read_path)
        if not read_path.exists():
            raise FileNotFoundError()
        ct_path = None
        seg_path = None
        
        for series_path in read_path.glob('*'):
            all_files = list(series_path.glob('*.dcm'))
            modality = pydicom.dcmread(all_files[0], stop_before_pixels=True).Modality

            if len(all_files) > 1 and modality == 'CT':
                if ct_path is None:
                    ct_path = series_path
                else:
                    raise DICOMDataAnomalyError('Multiple CT DICOM series were found.')
            elif len(all_files) == 1 and modality == 'SEG':
                if seg_path is None:
                    seg_path = series_path
                else:
                    raise DICOMDataAnomalyError('Multiple SEG DICOM series were found.')

        return ct_path, seg_path


class NSCLCRadiomicsReader(DICOMFileSystemReader):
    _CT_BACKEND = mt.LoadImage('ITKReader', image_only=True, ensure_channel_first=True, dtype=np.float32)
    _SEG_BACKEND = mt.LoadImage('PydicomReader', image_only=True, ensure_channel_first=True, dtype=np.float32)

    def __call__(self, *data, **params):
        ct_path, seg_path = self.find_series_paths(params['read_path'])

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


class NLSTReader(DICOMFileSystemReader):
    pass


class PAXReader(PipelinePart):
    pass

