from pathlib import Path
import os
import numpy as np
from .pipeline import PipelinePart


class NPZWriter(PipelinePart):
    def __init__(self, ct_save_path, seg_save_path):
        self._ct_save_path = ct_save_path
        self._seg_save_path = seg_save_path

    def __call__(self, *data, **params):
        ct_data, seg_data_list = data
        seg_data = seg_data_list[0]

        save_id = params.get('id', None)
        if save_id is None:
            raise Exception('No valid id was provided.')

        if ct_data is not None and self._ct_save_path is not None:
            ct_path = os.path.join(self._ct_save_path, save_id)
            Path(ct_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                ct_path,
                data=ct_data.numpy(), 
                affine=ct_data.affine.numpy(), 
                allow_pickle=False
            )
        if seg_data is not None and self._seg_save_path is not None:
            seg_path = os.path.join(self._seg_save_path, save_id)
            Path(seg_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                seg_path,
                data=seg_data.numpy(), 
                affine=seg_data.affine.numpy(), 
                allow_pickle=False
            )
        return (ct_data, seg_data), params

