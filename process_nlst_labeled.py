from ct_data_management.acquisition import IDCFileSystemDataManager, NLST_LABELED_INFO
from ct_data_management.processing.pipeline import PipelineStack
from ct_data_management.processing.readers import DICOMFileSystemReader, DICOMDataAnomalyError
from ct_data_management.processing.transforms import OrientTransform, ResampleTransform, ClipAndNormTransform, ToDeviceTransform
from ct_data_management.processing.writers import NPZWriter
from ct_data_management.processing.utils import InteractiveViewer, TimePipelinePart

import torch
import os
import hashlib
from tqdm import tqdm


def generate_uid(path):
    patient_id, studey_uid = str(path).split('/')[-3:-1]
    # 16 bit UID ~ 1 in 14 billion chanche for collision for 1 million datapoints
    uid = hashlib.shake_256((patient_id + studey_uid).encode()).hexdigest(8)
    return uid


if __name__ == '__main__':
    DATA_PATH = '/mnt/seagate_exp/radiology/data/raw/nlst_labeled'
    SAVE_PATH = '/mnt/seagate_exp/radiology/data/processed/nlst_labeled'

    torch.set_grad_enabled(False)  # No need for gradient calculation in this script

    #data_manager = IDCFileSystemDataManager(DATA_PATH, NLST_LABELED_INFO).sync_data()
    data_manager = IDCFileSystemDataManager(DATA_PATH, NLST_LABELED_INFO)

    pipeline = PipelineStack([
        DICOMFileSystemReader(lung_seg_labels=['lung'], nodule_seg_labels=['nodule']),
        OrientTransform(),
        ResampleTransform(),
        ClipAndNormTransform(clip_min=-1000, clip_max=400),
        NPZWriter(),
        #InteractiveViewer(),
    ])

    for ct_path, seg_path in tqdm(list(data_manager.get_paths())):
        new_uid = generate_uid(ct_path)
        try:
            pipeline(
                ct_path,
                seg_path,
                ct_save_path=f'ct/{new_uid}',
                seg_save_path=f'seg/{new_uid}',
                base_save_path=SAVE_PATH,
            )
        #except DICOMDataAnomalyError as e:
        except Exception as e:
            print('Error:', e)
            print('CT file:', ct_path)
            print('SEG file:', seg_path)
            print('Skipping files...')

