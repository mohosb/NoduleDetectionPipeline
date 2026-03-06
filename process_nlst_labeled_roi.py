from ct_data_management.acquisition import IDCFileSystemDataManager, NLST_LABELED_INFO
from ct_data_management.processing.pipeline import PipelineStack
from ct_data_management.processing.readers import DICOMFileSystemReader
from ct_data_management.processing.transforms import *
from ct_data_management.processing.writers import NPZWriter
from ct_data_management.processing.utils import InteractiveViewer, TimePipelinePart

import torch
import os
from tqdm import tqdm


if __name__ == '__main__':
    DATA_PATH = '/mnt/seagate_exp/radiology/data/raw/nlst_labeled'
    SAVE_PATH = '/mnt/seagate_exp/radiology/data/processed/nlst_labeled'

    torch.set_grad_enabled(False)  # No need for gradient calculation in this script

    #data_manager = IDCFileSystemDataManager(DATA_PATH, NLST_LABELED_INFO).sync_data()
    data_manager = IDCFileSystemDataManager(DATA_PATH, NLST_LABELED_INFO)

    pipeline = PipelineStack([
        DICOMFileSystemReader(return_headers=True, dtype=torch.float16),
        IDGenerator(),
        FilterSegmentsTransform(target_labels=['lung', 'nodule'], min_num_segments=2),
        OrientTransform(),
        ResampleTransform(),
        MergeSegmentsTransform(),
        ClipAndNormTransform(clip_min=-1000, clip_max=400),
        NPZWriter(os.path.join(SAVE_PATH, 'ct'), os.path.join(SAVE_PATH, 'roi_seg')),
        #InteractiveViewer(),
    ])

    for ct_path, seg_path_list in tqdm(list(data_manager.get_paths())):
        try:
            pipeline(ct_path, seg_path_list)
        except Exception as e:
            print('Error:', e)
            print('CT file:', ct_path)
            print('SEG file list:', seg_path_list)
            print('Skipping files...')

