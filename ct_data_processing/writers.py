from pathlib import Path
import numpy as np
from .pipeline import PipelinePart


class NPZWriter(PipelinePart):
    def __call__(self, *data, **params):
        ct_data, seg_data = data
        base_path = Path(params.get('base_save_path', ''))
        if ct_data is not None:
            ct_path = base_path.joinpath(params['ct_save_path'])
            ct_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                ct_path,
                data=ct_data.numpy(), 
                affine=ct_data.affine.numpy(), 
                allow_pickle=False
            )
        if seg_data is not None:
            seg_path = base_path.joinpath(params['seg_save_path'])
            seg_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                seg_path,
                data=seg_data.numpy(), 
                affine=seg_data.affine.numpy(), 
                allow_pickle=False
            )
        return (ct_data, seg_data), params

