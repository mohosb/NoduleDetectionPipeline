import torch
import monai.transforms as mt
import subprocess
import os
import pydicom
import shutil
from glob import glob
from .pipeline import PipelinePart
from ..utils import SmartTemporaryDirectory


class DICOMFileSystemReader(PipelinePart):
    def __init__(self, backend='NibabelReader', dtype=None, return_headers=True):
        self._backend = mt.LoadImage(backend, image_only=True, ensure_channel_first=True, dtype=dtype)
        self._return_headers = return_headers

    def __call__(self, *data, **params):
        ct_path, seg_path_list = data

        ct_data = None
        seg_data_list = None

        if ct_path is not None:
            if self._return_headers:
                params['ct_header'] = pydicom.dcmread(glob(os.path.join(ct_path, '*.dcm'))[0], stop_before_pixels=True)
                #params['ct_id'] = ct_path.split('/')[-1]

            ct_data = self._read_dicom(ct_path)

        if seg_path_list is not None:
            if self._return_headers:
                params['seg_header_list'] = [
                    pydicom.dcmread(glob(os.path.join(seg_path, '*.dcm'))[0], stop_before_pixels=True) for seg_path in seg_path_list
                ]
                #params['seg_id'] = seg_path.split('/')[-1]

            seg_data_list = [self._read_dicom(seg_path) for seg_path in seg_path_list] 

        return (ct_data, seg_data_list), params

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

    '''
    def _read_seg_dicom(self, path):
        required_space = shutil.disk_usage(path)[1] / 1024 ** 3
        with SmartTemporaryDirectory(required_space) as temp_dir:
            subprocess.run(
                [
                    'segimage2itkimage', 
                    '--inputDICOM', glob(os.path.join(path, '*.dcm'))[0], 
                    '--outputDirectory', temp_dir, 
                    '--outputType', 'nifti'
                ], 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE
            )
            data = self._backend(glob(os.path.join(temp_dir, '*.nii.gz')))
        return data
    '''
    
class PAXReader(PipelinePart):
    pass

