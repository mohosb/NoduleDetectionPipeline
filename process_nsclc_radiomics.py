from ct_data_processing.pipeline import PipelineStack
from ct_data_processing.readers import NSCLCRadiomicsReader
from ct_data_processing.transforms import OrientTransform, ResampleTransform, AutoCropTransform, ClipAndNormTransform
from ct_data_processing.writers import NPZWriter
from ct_data_processing.utils import InteractiveViewer, TimePipelinePart

import torch
import hashlib
from pathlib import Path
from tqdm import tqdm


def generate_uid(path):
    case_and_studey = '/'.join(str(path).split('/')[-2:])
    # 16 bit UID ~ 1 in 14 billion chanche for collision for 1 million datapoints
    uid = hashlib.shake_256(case_and_studey.encode()).hexdigest(8)
    return uid


if __name__ == '__main__':
    DATA_PATH = '/mnt/seagate_exp/radiology/data/raw_old/nsclc_radiomics'
    SAVE_PATH = '/mnt/seagate_exp/radiology/data/tmp/nsclc_radiomics'

    torch.set_grad_enabled(False)  # No need for gradient calculation in this script

    all_casestudy_paths = [p for p in Path(DATA_PATH).glob('*/*') if p.is_dir()]

    pipeline = PipelineStack([
        NSCLCRadiomicsReader(),
        OrientTransform(),
        #ResampleTransform(),
        ClipAndNormTransform(clip_min=-1000, clip_max=400),
        AutoCropTransform(scale_factor=32, threshold=0.7, padding=0),
        #NPZWriter(),
        InteractiveViewer(),
    ])

    for casestudy_path in tqdm(all_casestudy_paths):
        uid = generate_uid(casestudy_path)
        pipeline(
            read_path=casestudy_path, 
            ct_save_path=f'ct/{uid}',
            seg_save_path=f'seg/{uid}',
            base_save_path=SAVE_PATH,
        )
        exit()

