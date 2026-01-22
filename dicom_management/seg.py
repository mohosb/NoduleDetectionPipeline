import torch
import numpy as np
import pydicom
from pathlib import Path
import monai.transforms as mt
import os
from tqdm import tqdm
from .utils import DICOMDataAnomalyError, generate_uid


_DATA_READER = mt.Compose([
    mt.LoadImage('PydicomReader', image_only=True, ensure_channel_first=True, dtype=np.float16),
    mt.Orientation(axcodes='RAS', labels=None),
])
_DATA_RESAMPLER = mt.Spacing(pixdim=(1., 1., 1.), mode='nearest')

def find_series(casestudy_path):
    for series_path in casestudy_path.glob('*'):
        all_files = list(series_path.glob('*.dcm'))
        modality = pydicom.dcmread(all_files[0], stop_before_pixels=True).Modality

        if len(all_files) == 1 and modality == 'SEG':
            return series_path

    raise DICOMDataAnomalyError('No SEG DICOM file was found.')

def read_data(series_path):
    return _DATA_READER(series_path)

def process_data(data):
    data = _DATA_RESAMPLER(data)
    
    # NSCLC-Radiomics has the following labels:
    #{'Spinal-Cord', 'Lung-Right', 'Lung-Left', 'Heart', 'Esophagus', 'Lungs-Total', 'GTV-1'}
    lung_segments = []
    nodule_segments = []
    for seg_label, seg_number in data.meta['labels'].items():
        seg_label = seg_label.strip().lower()
        if 'lung' in seg_label:
            lung_segments.append(seg_number)
        elif seg_label == 'gtv-1':
            nodule_segments.append(seg_number)

    if len(lung_segments) == 0:
        raise DICOMDataAnomalyError('No Lung instance could be detected in SEG DICOM file.')
    
    # Union Region of Interest instances into a single instance and separate the nodule(s)
    roi_data = data[lung_segments + nodule_segments].sum(0).clamp_(0, 1)
    nodule_data = data[nodule_segments].sum(0).clamp_(0, 1)
    data = torch.stack((roi_data, nodule_data)).to(torch.uint8)

    return data

def write_data(data, path):
    np.savez_compressed(path, data=data.numpy(), affine=data.affine.numpy(), allow_pickle=False)

def run_pipeline(data_path, save_path, allow_fail=True):
    try:
        uid = generate_uid(data_path)
        data = read_data(find_series(data_path))
        data = process_data(data)
        write_data(data, os.path.join(save_path, uid))
    except DICOMDataAnomalyError as e:
        if allow_fail:
            print(f'Error processing {str(data_path)}: {str(e)}')
        else:
            raise e


if __name__ == '__main__':
    DATA_PATH = '/mnt/seagate_exp/radiology/data/raw/nsclc_radiomics'
    #DATA_PATH = '/mnt/seagate_exp/radiology/data/raw/nlst_labeled'
    SAVE_PATH = '/mnt/seagate_exp/radiology/data/processed/nsclc_radiomics'

    torch.set_grad_enabled(False)  # No need for gradient calculation in this script

    os.makedirs(SAVE_PATH, exist_ok=True)
    all_casestudy_paths = [p for p in Path(DATA_PATH).glob('*/*') if p.is_dir()]
    #all_casestudy_paths = all_casestudy_paths[:10]

    for casestudy_path in tqdm(all_casestudy_paths):
        run_pipeline(casestudy_path, SAVE_PATH)

