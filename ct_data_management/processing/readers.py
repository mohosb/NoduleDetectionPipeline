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

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_path      = data.get('ct')
        seg_path_list = data.get('seg_list')

        if ct_path is not None:
            if self._return_headers:
                params['ct_header'] = pydicom.dcmread(self._find_dicom(ct_path), stop_before_pixels=True)
                #params['ct_id'] = ct_path.split('/')[-1]
            data['ct'] = self._read_dicom(ct_path)

        if seg_path_list is not None:
            if self._return_headers:
                params['seg_header_list'] = [
                    pydicom.dcmread(self._find_dicom(p), stop_before_pixels=True)
                    for p in seg_path_list
                ]
                #params['seg_id'] = seg_path.split('/')[-1]
            data['seg_list'] = [self._read_dicom(p) for p in seg_path_list]

        return data, params

    @staticmethod
    def _find_dicom(path):
        """Return the path to one DICOM file in a directory, or raise a clear error."""
        dcm_files = glob(os.path.join(path, '*.dcm'))
        if not dcm_files:
            raise FileNotFoundError(f'No DICOM files found in: {path}')
        return dcm_files[0]

    def _read_dicom(self, path):
        # Use the actual size of the DICOM files as the space estimate for the
        # temporary NIfTI conversion, not the filesystem-wide usage figure.
        required_space = sum(
            os.path.getsize(f) for f in glob(os.path.join(path, '*.dcm'))
        )
        with SmartTemporaryDirectory(required_space) as temp_dir:
            subprocess.run(
                ['dcm2niix', '-a', 'y', '-z', 'n', '-o', temp_dir, '-f', '_temp_dcm2niix_file', str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            nii_path = os.path.join(temp_dir, '_temp_dcm2niix_file.nii')
            if not os.path.exists(nii_path):
                raise FileNotFoundError(
                    f'dcm2niix exited successfully but produced no output for: {path}'
                )
            data = self._backend(nii_path)
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
