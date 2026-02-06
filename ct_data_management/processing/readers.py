import torch
import monai.transforms as mt
import numpy as np
import tempfile
import subprocess
import os
import pydicom
import shutil
from glob import glob
from .pipeline import PipelinePart
from ..utils import SmartTemporaryDirectory


class DICOMDataAnomalyError(Exception):
    pass


class NSCLCRadiomicsReader(PipelinePart):
    def __init__(self, lung_seg_labels=tuple(), nodule_seg_labels=tuple(), backend='ITKReader', dtype=np.float32):
        self._backend = mt.LoadImage(backend, image_only=True, ensure_channel_first=True, dtype=dtype)
        self._lung_seg_labels = lung_seg_labels
        self._nodule_seg_labels = nodule_seg_labels

    def __call__(self, *data, **params):
        ct_path, seg_path = data

        ct_data = None
        seg_data = None

        if ct_path is not None:
            ct_data = self._read_dicom(ct_path)

        if seg_path is not None:
            seg_data = self._read_dicom(seg_path)

            seg_header = pydicom.dcmread(glob(os.path.join(seg_path, '*.dcm'))[0], stop_before_pixels=True)
            lung_segments = set()
            nodule_segments = set()
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
                    for label_pattern in self._lung_seg_labels:
                        if label_pattern in seg_label:
                            lung_segments.add(seg_num)
                    for label_pattern in self._nodule_seg_labels:
                        if label_pattern in seg_label:
                            nodule_segments.add(seg_num)

            if len(lung_segments) == 0:
                raise DICOMDataAnomalyError('No Lung instance could be detected in SEG DICOM file.')
    
            # Union Region of Interest instances into a single instance and separate the nodule(s)
            roi_data = seg_data[list(lung_segments.union(nodule_segments))].sum(0).clamp_(0, 1)
            nodule_data = seg_data[list(nodule_segments)].sum(0).clamp_(0, 1)
            seg_data = torch.stack((roi_data, nodule_data))

        return (ct_data, seg_data), params

    def _read_dicom(self, path):
        required_space = shutil.disk_usage(path)[1] / 1024 ** 3
        with SmartTemporaryDirectory(required_space) as temp_dir:
            subprocess.run(
                ['dcm2niix', '-a', 'y', '-z', 'n', '-o', temp_dir, '-f', '_temp_dcm2niix_file', str(path)], 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE
            )
            data = self._backend(os.path.join(temp_dir, '_temp_dcm2niix_file.nii'))
        return data


class NLSTReader(PipelinePart):
    pass


class LIDCIDRIReader(PipelinePart):
    pass


class PAXReader(PipelinePart):
    pass

