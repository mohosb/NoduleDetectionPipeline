import torch
import numpy as np
import pydicom
from pathlib import Path
import monai.transforms as mt
import os
from .utils import DICOMDataAnomalyError, generate_uid


_DATA_READER = mt.Compose([
    mt.LoadImage('ITKReader', image_only=True, ensure_channel_first=True, dtype=np.float16),
    mt.Orientation(axcodes='RAS', labels=None),
])
_DATA_RESAMPLER = mt.Spacing(pixdim=(1., 1., 1.), mode='bilinear')

def find_series(casestudy_path):
    for series_path in casestudy_path.glob('*'):
        all_files = list(series_path.glob('*.dcm'))
        modality = pydicom.dcmread(all_files[0], stop_before_pixels=True).Modality

        if len(all_files) > 1 and modality == 'CT':
            return series_path

    raise DICOMDataAnomalyError('No CT DICOM file was found.')

def read_data(series_path):
    return _DATA_READER(series_path)

def process_data(data):
    return _DATA_RESAMPLER(data).round_().to(torch.int16)

def write_data(data, path):
    np.savez_compressed(path, data=data.numpy(), affine=data.affine.numpy(), allow_pickle=False)

def run_pipeline(data_path, save_path, allow_fail=True):
    try:
        uid = generate_uid(data_path)
        write_data(
            process_data(
                read_data(
                    find_series(data_path)
                )
            ),
            os.path.join(save_path, uid)
        )
    except DICOMDataAnomalyError as e:
        if allow_fail:
            print(f'Error processing {str(data_path)}: {str(e)}')
        else:
            raise e


if __name__ == '__main__':
    DATA_PATH = '/mnt/seagate_exp/radiology/data/raw/nsclc_radiomics'
    SAVE_PATH = '/mnt/seagate_exp/radiology/data/processed/nsclc_radiomics'

    torch.set_grad_enabled(False)  # No need for gradient calculation in this script

    os.makedirs(SAVE_PATH, exist_ok=True)
    all_casestudy_paths = [p for p in Path(DATA_PATH).glob('*/*') if p.is_dir()]
    #all_casestudy_paths = all_casestudy_paths[:10]

    for casestudy_path in all_casestudy_paths:
        run_pipeline(DATA_PATH, SAVE_PATH)

