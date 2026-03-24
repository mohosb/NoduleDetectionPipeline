from pathlib import Path
import os
import numpy as np
from .pipeline import PipelinePart


class NPZWriter(PipelinePart):
    """Save CT and segmentation data as NPZ files.

    Args:
        ct_save_path:  Directory to write CT files into.
        seg_save_path: Directory to write segmentation files into.
        save_mode:     '3d' — one NPZ per volume (default):
                              <save_path>/<series_id>.npz
                       '2d' — one NPZ per axial slice (last axis, D in C×H×W×D):
                              <save_path>/<series_id>_<slice_idx:04d>.npz
                       The original 3D affine is stored in every 2D slice file
                       as a spatial reference.
        compress:      If True (default), use np.savez_compressed.
                       If False, use np.savez (faster writes, larger files).
    """

    def __init__(self, ct_save_path, seg_save_path, save_mode='3d', compress=True):
        if save_mode not in ('3d', '2d'):
            raise ValueError(f"save_mode must be '3d' or '2d', got '{save_mode}'")
        self._ct_save_path  = ct_save_path
        self._seg_save_path = seg_save_path
        self._save_mode     = save_mode
        self._savez         = np.savez_compressed if compress else np.savez

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_data  = data.get('ct')
        seg_data = data.get('seg')

        save_id = params.get('id', None)
        if save_id is None:
            raise ValueError('No valid id was provided. Ensure IDGenerator runs before NPZWriter.')

        if self._save_mode == '3d':
            self._save_3d(ct_data, seg_data, save_id)
        else:
            self._save_2d(ct_data, seg_data, save_id)

        return data, params

    # --- Save modes ---

    def _save_3d(self, ct_data, seg_data, save_id):
        if ct_data is not None and self._ct_save_path is not None:
            self._write(self._ct_save_path, save_id, ct_data.cpu().numpy(), ct_data.affine.cpu().numpy())
        if seg_data is not None and self._seg_save_path is not None:
            self._write(self._seg_save_path, save_id, seg_data.cpu().numpy(), seg_data.affine.cpu().numpy())

    def _save_2d(self, ct_data, seg_data, save_id):
        # Tensor layout after pipeline: (C, H, W, D); slices along last axis.
        n_slices = (ct_data if ct_data is not None else seg_data).shape[-1]

        # Preserve the 3D affine in every slice file as a spatial reference.
        affine = (ct_data if ct_data is not None else seg_data).affine.cpu().numpy()

        for i in range(n_slices):
            slice_id = f'{save_id}_{i:04d}'
            if ct_data is not None and self._ct_save_path is not None:
                self._write(self._ct_save_path, slice_id, ct_data[..., i].cpu().numpy(), affine)
            if seg_data is not None and self._seg_save_path is not None:
                self._write(self._seg_save_path, slice_id, seg_data[..., i].cpu().numpy(), affine)

    def _write(self, save_dir, file_id, data_arr, affine_arr):
        path = os.path.join(save_dir, file_id)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._savez(path, data=data_arr, affine=affine_arr, allow_pickle=False)
