import numpy as np
import torch
import torch.nn.functional as F
import monai.transforms as mt
import scipy.ndimage as ndi
from monai.data import MetaTensor
from monai.transforms import SpatialCrop
from .pipeline import PipelinePart


class DataAnomalyError(Exception):
    pass


class IDGenerator(PipelinePart):
    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_header = params.get('ct_header', None)
        if ct_header is not None:
            params['id'] = ct_header.SeriesInstanceUID
        return data, params


class FilterSegmentsTransform(PipelinePart):
    """Filter DICOM segmentation files into separate nodule and lung mask lists.

    For each SEG file the transform runs two independent label-matching passes and
    accumulates results into ``data['nodule_seg_list']`` and ``data['lung_seg_list']``
    respectively.  A label set of ``None`` means "do not load this type".

    Raises ``DataAnomalyError`` after processing all files if a non-None label set
    produced no matching segments across any file.
    """

    def __init__(self, nodule_labels=None, lung_labels=None):
        self._nodule_labels = list(nodule_labels) if nodule_labels else None
        self._lung_labels   = list(lung_labels)   if lung_labels   else None

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        seg_header_list = params.get('seg_header_list', [])
        seg_data_list   = data.pop('seg_list', [])   # raw reader field; consumed here

        nodule_out, lung_out = [], []

        for seg_header, seg_data in zip(seg_header_list, seg_data_list):
            seg_map = self._build_segment_map(seg_header)

            if self._nodule_labels is not None:
                mask = self._extract(seg_data, seg_map, self._nodule_labels)
                if mask is not None:
                    nodule_out.append(mask)

            if self._lung_labels is not None:
                mask = self._extract(seg_data, seg_map, self._lung_labels)
                if mask is not None:
                    lung_out.append(mask)

        if self._nodule_labels is not None and not nodule_out:
            raise DataAnomalyError('No nodule segmentation found.')
        if self._lung_labels is not None and not lung_out:
            raise DataAnomalyError('No lung segmentation found.')

        if nodule_out:
            data['nodule_seg_list'] = nodule_out
        if lung_out:
            data['lung_seg_list'] = lung_out

        return data, params

    @staticmethod
    def _build_segment_map(seg_header):
        """Return {segment_index: lowercase_label} from DICOM SegmentSequence."""
        m = {}
        if 'SegmentSequence' in seg_header:
            for item in seg_header.SegmentSequence:
                idx   = item.SegmentNumber - 1
                label = (getattr(item, 'SegmentLabel', None)
                         or getattr(item, 'SegmentDescription', '') or '').lower()
                m[idx] = label
        return m

    @staticmethod
    def _extract(seg_data, seg_map, label_patterns):
        """Sum channels whose label matches any pattern; return None if none match."""
        indices = [i for i, lbl in seg_map.items()
                   if any(p in lbl for p in label_patterns)
                   and i < seg_data.size(0)]
        if not indices:
            return None
        return seg_data[indices].sum(0, keepdim=True).clamp_(0, 1)


class HUClipAndNormTransform(PipelinePart):
    def __init__(self, clip_min, clip_max):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        ct_data = data.get('ct')

        if ct_data is not None:
            ct_data.clip_(self.clip_min, self.clip_max)
            data['ct'] = (ct_data - self.clip_min) / (self.clip_max - self.clip_min)

        return data, params


class OrientTransform(PipelinePart):
    def __init__(self, orientation='RAS'):
        self._backend = mt.Orientation(axcodes=orientation, labels=None)

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        if data.get('ct') is not None:
            data['ct'] = self._backend(data['ct'])
        for key in ('nodule_seg_list', 'lung_seg_list'):
            if data.get(key):
                data[key] = [self._backend(s) for s in data[key]]
        return data, params


class ResampleTransform(PipelinePart):
    def __init__(self, spacing=(1., 1., 1.), ct_mode='bilinear', seg_mode='nearest'):
        self._ct_backend  = mt.Spacing(pixdim=spacing, mode=ct_mode)
        self._seg_backend = mt.ResampleToMatch(mode=seg_mode, padding_mode='zeros')

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        if data.get('ct') is not None:
            data['ct'] = self._ct_backend(data['ct'])
        for key in ('nodule_seg_list', 'lung_seg_list'):
            if data.get(key):
                data[key] = [self._seg_backend(s, data['ct']) for s in data[key]]
        return data, params


class LungDilationTransform(PipelinePart):
    """Expand the lung mask outward by iterating binary dilation.

    Operates on ``data['lung_seg_list']`` only.  The nodule masks are untouched.
    When this runs before ``ROICropTransform`` the crop bounding box (computed from
    dilated_lung ∪ nodule) will include a margin around the pleural boundary,
    covering nodules that protrude past the lung annotation.

    Uses 26-connectivity so the expansion is isotropic in all directions.
    """

    _STRUCTURE = ndi.generate_binary_structure(3, 3)   # 26-connectivity

    def __init__(self, n_dilations: int):
        self._n = n_dilations

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        lung_segs = data.get('lung_seg_list')
        if not lung_segs:
            return data, params
        data['lung_seg_list'] = [
            MetaTensor(
                torch.from_numpy(
                    ndi.binary_dilation(
                        seg[0].cpu().numpy().astype(bool),
                        structure=self._STRUCTURE,
                        iterations=self._n,
                    )
                ).unsqueeze(0).to(seg.dtype),
                meta=seg.meta,
            )
            for seg in lung_segs
        ]
        return data, params


class ROICropTransform(PipelinePart):
    """Crop CT and all segmentation masks to a bounding box around the chest.

    The bounding box is derived from the **temporary union** of all lung masks
    (possibly dilated by a preceding ``LungDilationTransform``) and all nodule
    masks.  This ensures that:

    - Nodules excluded from the lung annotation are still inside the crop.
    - Nodules that protrude past the pleural surface are fully retained.

    The union is not stored; ``lung_seg_list`` and ``nodule_seg_list`` remain
    separate after cropping.  MONAI's ``SpatialCrop`` updates each tensor's
    affine automatically, so downstream writers produce geometrically correct
    output.
    """

    def __init__(self, padding: int = 1):
        self._padding = padding

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        if data.get('ct') is None:
            return data, params

        # Gather all lung + nodule masks as the bounding-box reference.
        ref_segs = []
        for key in ('lung_seg_list', 'nodule_seg_list'):
            if data.get(key):
                ref_segs.extend(data[key])

        if not ref_segs:
            return data, params

        shape = ref_segs[0][0].shape   # (H, W, D)
        union = np.zeros(shape, dtype=bool)
        for seg in ref_segs:
            union |= seg[0].cpu().numpy().astype(bool)

        if not union.any():
            raise DataAnomalyError('Combined lung + nodule mask is empty; cannot compute crop.')

        roi_start, roi_end = [], []
        for ax in range(3):
            nax = tuple(j for j in range(3) if j != ax)
            idx = np.where(union.any(axis=nax))[0]
            roi_start.append(max(0,         int(idx.min()) - self._padding))
            roi_end.append(  min(shape[ax], int(idx.max()) + 1 + self._padding))

        cropper = SpatialCrop(roi_start=roi_start, roi_end=roi_end)
        data['ct'] = cropper(data['ct'])
        for key in ('nodule_seg_list', 'lung_seg_list'):
            if data.get(key):
                data[key] = [cropper(s) for s in data[key]]

        return data, params


class NoduleAnomalyFilterTransform(PipelinePart):
    """Remove nodule annotation components outside plausible size and HU ranges.

    Operates on ``data['nodule_seg_list']`` before the masks are merged.  For each
    mask entry the transform runs 3D connected-component labelling (26-connectivity)
    and independently checks each component against two criteria:

    1. **Volume** — component voxel count must be in ``[min_voxels, max_voxels]``.
       After resampling to 1 × 1 × 1 mm, voxel count ≈ volume in mm³.
    2. **Mean HU** — mean CT value of voxels in the component must be in
       ``[min_hu, max_hu]``.  This check uses ``data['ct']`` which still holds raw
       HU values at this stage (before ``HUClipAndNormTransform``).

    Raises ``DataAnomalyError`` if every component across all entries is removed,
    since empty output files are not acceptable.

    Default thresholds (adjust after medical expert review):
        min_voxels  =    14.0   ≈ 3 mm Ø sphere (Fleischner lower bound)
        max_voxels  = 14137.0   ≈ 30 mm Ø sphere (Fleischner nodule–mass boundary)
        min_hu      =  -800.0   below ground-glass range; excludes air artefacts
        max_hu      =   700.0   above calcified nodules; excludes dense cortical bone
    """

    _STRUCTURE = ndi.generate_binary_structure(3, 3)   # 26-connectivity

    def __init__(self, min_voxels=float('-inf'), max_voxels=float('inf'),
                 min_hu=float('-inf'), max_hu=float('inf')):
        self._min_vol = min_voxels
        self._max_vol = max_voxels
        self._min_hu  = min_hu
        self._max_hu  = max_hu

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        seg_list = data.get('nodule_seg_list')
        if not seg_list:
            return data, params

        ct_np = (data['ct'][0].cpu().numpy().astype(np.float32)
                 if data.get('ct') is not None else None)

        filtered = []
        total_components = 0
        for seg in seg_list:
            mask       = seg[0].cpu().numpy().astype(bool)
            labeled, n = ndi.label(mask, structure=self._STRUCTURE)
            total_components += n
            clean      = np.zeros_like(mask)
            for lbl in range(1, n + 1):
                comp = labeled == lbl
                if not (self._min_vol <= int(comp.sum()) <= self._max_vol):
                    continue
                if ct_np is not None:
                    if not (self._min_hu <= float(ct_np[comp].mean()) <= self._max_hu):
                        continue
                clean |= comp
            filtered.append(MetaTensor(
                torch.from_numpy(clean).unsqueeze(0).to(seg.dtype),
                meta=seg.meta,
            ))

        if all(not f[0].any() for f in filtered):
            if total_components == 0:
                raise DataAnomalyError(
                    'Nodule segmentation mask is empty (no annotated voxels). Series dropped.'
                )
            raise DataAnomalyError(
                f'All {total_components} nodule component(s) removed by anomaly filter '
                '(volume or mean HU out of range). Series dropped.'
            )

        data['nodule_seg_list'] = filtered
        return data, params


class MergeSegmentsTransform(PipelinePart):
    """Merge per-file segmentation lists into single tensors.

    Consumes ``data['nodule_seg_list']`` and ``data['lung_seg_list']`` and
    produces ``data['nodule_seg']`` and ``data['lung_seg']`` respectively.
    Lists with a single entry are used directly; multiple entries are summed
    (binary OR) and clamped to [0, 1].
    """

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        for list_key, seg_key in (('nodule_seg_list', 'nodule_seg'),
                                   ('lung_seg_list',   'lung_seg')):
            lst = data.pop(list_key, None)
            if not lst:
                data[seg_key] = None
            elif len(lst) == 1:
                data[seg_key] = lst[0]
            else:
                data[seg_key] = torch.stack(lst, dim=0).sum(dim=0).clamp_(0, 1)
        return data, params


class ComputeROITransform(PipelinePart):
    """Compute the ROI mask as the union of the lung and nodule segmentations.

    This is the only point in the pipeline where a merged lung+nodule mask
    exists.  The result is stored in ``data['roi_seg']`` and is only added to
    the pipeline when ``roi`` is included in ``--output``.

    Must run before ``NoduleInstanceSegTransform`` so that ``roi_seg`` is a
    clean binary mask, not a mix of instance labels and binary lung values.
    """

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        lung   = data.get('lung_seg')
        nodule = data.get('nodule_seg')

        if lung is None and nodule is None:
            raise DataAnomalyError('Cannot compute ROI: both lung_seg and nodule_seg are absent.')

        if lung is None:
            data['roi_seg'] = nodule
        elif nodule is None:
            data['roi_seg'] = lung
        else:
            roi = (lung + nodule).clamp_(0, 1)
            data['roi_seg'] = MetaTensor(roi, meta=lung.meta)

        return data, params


class ToDeviceTransform(PipelinePart):
    def __init__(self, device='cpu'):
        self._device = torch.device(device)

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        for key in ('ct', 'nodule_seg', 'lung_seg', 'roi_seg'):
            if data.get(key) is not None:
                data[key] = data[key].to(self._device)
        for key in ('nodule_seg_list', 'lung_seg_list'):
            if data.get(key):
                data[key] = [s.to(self._device) for s in data[key]]
        return data, params


class NoduleInstanceSegTransform(PipelinePart):
    """Convert a binary nodule segmentation mask into an instance segmentation mask.

    Runs 3D connected-component analysis (26-connectivity) on ``data['nodule_seg']``
    so that each spatially distinct nodule receives a unique positive integer
    label. Background remains 0.  Two voxels that touch — even diagonally —
    are treated as belonging to the same nodule, matching the assumption that
    touching nodules in 3D CT are a single lesion.

    Must be placed after ``MergeSegmentsTransform`` (which produces
    ``data['nodule_seg']``) and after ``ComputeROITransform`` (so that
    ``roi_seg`` is a binary union, not mixed with instance labels).
    """

    _STRUCTURE = ndi.generate_binary_structure(3, 3)   # 26-connectivity

    def __call__(self, data: dict, params: dict) -> tuple[dict, dict]:
        seg = data.get('nodule_seg')
        if seg is None:
            return data, params

        mask_np = seg[0].cpu().numpy().astype(bool)
        labeled_np, _ = ndi.label(mask_np, structure=self._STRUCTURE)

        # int16 supports up to 32 767 instances per volume, which is ample.
        labeled_tensor = torch.from_numpy(labeled_np).unsqueeze(0).to(torch.int16)
        data['nodule_seg'] = MetaTensor(labeled_tensor, meta=seg.meta)

        return data, params
