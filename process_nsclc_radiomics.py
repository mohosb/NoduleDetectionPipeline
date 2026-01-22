import torch
import os
import dicom_management as dcmm
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    DATA_PATH = '/mnt/seagate_exp/radiology/data/raw/nsclc_radiomics'
    SAVE_PATH = '/mnt/seagate_exp/radiology/data/processed/nsclc_radiomics'

    torch.set_grad_enabled(False)  # No need for gradient calculation in this script

    os.makedirs(SAVE_PATH, exist_ok=True)
    all_casestudy_paths = [p for p in Path(DATA_PATH).glob('*/*') if p.is_dir()]

    # Test on a single case:
    #data_path = all_casestudy_paths[0]
    #ct_data = dcmm.ct.read_data(dcmm.ct.find_series(data_path))
    #ct_data = dcmm.ct.process_data(ct_data)
    #seg_data = dcmm.seg.read_data(dcmm.seg.find_series(data_path))
    #seg_data = dcmm.seg.process_data(seg_data)
    #dcmm.utils.visualize(ct_data, seg_data)
    #exit()

    for casestudy_path in tqdm(all_casestudy_paths):
        dcmm.ct.run_pipeline(DATA_PATH, SAVE_PATH)
        dcmm.seg.run_pipeline(DATA_PATH, SAVE_PATH)

