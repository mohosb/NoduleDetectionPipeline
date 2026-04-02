'''
Unified processing entry point for all CT datasets.

Usage:
    python process.py --dataset lidc_idri       --mode nodule
    python process.py --dataset nsclc_radiomics --mode nodule
    python process.py --dataset nsclc_radiomics --mode roi
    python process.py --dataset nlst_labeled    --mode nodule
    python process.py --dataset nlst_labeled    --mode roi
    python process.py --dataset lidc_idri       --mode nodule --sync          # also downloads data
    python process.py --dataset lidc_idri       --mode nodule --download-only # download raw data only, skip processing
    python process.py --dataset lidc_idri       --mode nodule --reprocess     # ignore already-processed entries
    python process.py --dataset lidc_idri       --mode nodule --save-mode 2d           # save per-slice
    python process.py --dataset lidc_idri       --mode nodule --save-mode 3d,2d        # save both 3D and per-slice
    python process.py --dataset lidc_idri       --mode nodule --workers 4              # use 4 GPU workers
    python process.py --dataset lidc_idri       --mode nodule --cpu                    # force CPU single-process
    python process.py --dataset lidc_idri       --mode nodule --view                   # interactive viewer instead of saving
    python process.py --dataset lidc_idri       --mode nodule --label-type instance              # instance segmentation masks
    python process.py --dataset lidc_idri       --mode nodule --label-type semantic,instance     # both semantic and instance masks
    python process.py --dataset lidc_idri       --mode nodule --save-format nifti      # save as NIfTI instead of NumPy
    python process.py --dataset lidc_idri       --mode nodule --save-format numpy,nifti  # save as both NumPy and NIfTI
    python process.py --dataset lidc_idri       --mode nodule --save-format nifti --no-compress  # uncompressed .nii
'''

import argparse
import logging
import logging.handlers
import multiprocessing as mp
import os
import traceback
from datetime import datetime
from multiprocessing import current_process

import torch
from tqdm import tqdm

from ct_data_management.acquisition import (
    IDCFileSystemDataManager,
    LIDC_IDRI_INFO,
    NSCLC_RADIOMICS_INFO,
    NLST_LABELED_INFO,
)
from ct_data_management.processing.pipeline import PipelineStack
from ct_data_management.processing.readers import DICOMFileSystemReader
from ct_data_management.processing.transforms import (
    IDGenerator,
    FilterSegmentsTransform,
    OrientTransform,
    ResampleTransform,
    MergeSegmentsTransform,
    ClipAndNormTransform,
    ToDeviceTransform,
    NoduleInstanceSegTransform,
)
from ct_data_management.processing.writers import NPZWriter, NIfTIWriter
from ct_data_management.processing.utils import InteractiveViewer, TimePipelinePart


RAW_BASE  = '/mnt/seagate_exp/radiology/data/raw'
PROC_BASE = '/mnt/seagate_exp/radiology/data/processed'

# Per-dataset configuration: (DatasetConfig, raw_subdir, processed_subdir)
DATASET_CONFIGS = {
    'lidc_idri':       LIDC_IDRI_INFO,
    'nsclc_radiomics': NSCLC_RADIOMICS_INFO,
    'nlst_labeled':    NLST_LABELED_INFO,
}

# Per-mode configuration: (target_labels, min_num_segments, seg_base)
# seg_base is used to construct output directory names:
#   semantic  → {seg_base}_sem_seg_{save_mode}   (e.g. nodule_sem_seg_3d)
#   instance  → {seg_base}_inst_seg_{save_mode}  (e.g. nodule_inst_seg_3d)
MODE_CONFIGS = {
    'nodule': {
        'lidc_idri':       (['nodule'],           1, 'nodule'),
        'nsclc_radiomics': (['neoplasm'],         1, 'nodule'),
        'nlst_labeled':    (['nodule'],           1, 'nodule'),
    },
    'roi': {
        'nsclc_radiomics': (['lung', 'neoplasm'], 2, 'roi'),
        'nlst_labeled':    (['lung', 'nodule'],   2, 'roi'),
    },
}

# --- Worker process state (process-local globals) ---

_worker_pipeline  = None
_worker_log_queue = None


def _worker_init(n_gpus, log_queue, pipeline_kwargs):
    global _worker_pipeline, _worker_log_queue

    worker_idx = current_process()._identity[0] - 1  # 0-based
    device = f'cuda:{worker_idx % n_gpus}' if n_gpus > 0 else 'cpu'

    _worker_log_queue = log_queue
    _worker_pipeline  = build_pipeline(device=device, **pipeline_kwargs)

    # Route all log records in this worker to the shared queue in the main process.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(log_queue))


def _process_series(args):
    ct_path, seg_path_list = args
    series_id = os.path.basename(ct_path)
    logger = logging.getLogger('pipeline')
    try:
        _worker_pipeline({'ct': ct_path, 'seg_list': seg_path_list}, {})
        logger.debug(f'OK: {series_id}')
        return {'series_id': series_id, 'ok': True}
    except Exception as e:
        logger.error(f'FAILED: {series_id}')
        logger.error(f'  CT:   {ct_path}')
        logger.error(f'  SEG:  {seg_path_list}')
        logger.error(f'  {type(e).__name__}: {e}')
        logger.debug(traceback.format_exc())
        return {'series_id': series_id, 'ok': False, 'error': str(e)}


# --- Pipeline builder ---

def build_pipeline(save_path: str, target_labels: list, min_num_segments: int,
                   seg_base: str, save_modes: list = None, save_formats: list = None,
                   compress: bool = True, device: str = 'cpu', viewer: bool = False,
                   label_types: list = None):
    if save_modes is None:
        save_modes = ['3d']
    if save_formats is None:
        save_formats = ['numpy']
    if label_types is None:
        label_types = ['semantic']

    parts = [DICOMFileSystemReader(return_headers=True, dtype=torch.float16)]

    if device != 'cpu':
        parts.append(ToDeviceTransform(device=device))

    parts += [
        IDGenerator(),
        FilterSegmentsTransform(target_labels=target_labels, min_num_segments=min_num_segments),
        OrientTransform(),
        ResampleTransform(),
        MergeSegmentsTransform(),
        ClipAndNormTransform(clip_min=-1000, clip_max=400),
    ]

    if device != 'cpu':
        parts.append(ToDeviceTransform(device='cpu'))

    if viewer:
        parts.append(InteractiveViewer())
        return PipelineStack(parts)

    has_semantic = 'semantic' in label_types
    has_instance = 'instance' in label_types

    if has_semantic:
        for save_mode in save_modes:
            ct_dir  = os.path.join(save_path, f'ct_{save_mode}')
            seg_dir = os.path.join(save_path, f'{seg_base}_sem_seg_{save_mode}')
            for save_format in save_formats:
                if save_format == 'nifti':
                    parts.append(NIfTIWriter(ct_dir, seg_dir, save_mode=save_mode, compress=compress))
                else:
                    parts.append(NPZWriter(ct_dir, seg_dir, save_mode=save_mode, compress=compress))

    if has_instance:
        parts.append(NoduleInstanceSegTransform())
        for save_mode in save_modes:
            # CT already written by semantic writers; skip it to avoid redundant I/O.
            ct_dir  = None if has_semantic else os.path.join(save_path, f'ct_{save_mode}')
            seg_dir = os.path.join(save_path, f'{seg_base}_inst_seg_{save_mode}')
            for save_format in save_formats:
                if save_format == 'nifti':
                    parts.append(NIfTIWriter(ct_dir, seg_dir, save_mode=save_mode, compress=compress))
                else:
                    parts.append(NPZWriter(ct_dir, seg_dir, save_mode=save_mode, compress=compress))

    return PipelineStack(parts)


# --- Logging ---

class _TqdmLoggingHandler(logging.StreamHandler):
    """Routes console log output through tqdm.write() to avoid disrupting the progress bar."""
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(log_dir: str, dataset: str, mode: str, log_queue=None):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{dataset}_{mode}_{timestamp}.log')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s'))

    ch = _TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))

    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # prevent records reaching the root logger's handlers (e.g. installed by torch/monai)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'Logging to {log_file}')

    if log_queue is not None:
        # Workers route records to this queue; listener dispatches them to fh and ch.
        listener = logging.handlers.QueueListener(log_queue, fh, ch, respect_handler_level=True)
        return logger, listener

    return logger, None


# --- Helpers ---

def already_processed(save_path: str, seg_base: str, series_id: str,
                      save_modes: list, save_formats: list, label_types: list,
                      compress: bool) -> bool:
    for save_mode in save_modes:
        ct_dir = os.path.join(save_path, f'ct_{save_mode}')
        for save_format in save_formats:
            ext = '.nii.gz' if (save_format == 'nifti' and compress) else \
                  '.nii'    if  save_format == 'nifti'               else '.npz'
            stem = series_id if save_mode == '3d' else f'{series_id}_0000'
            # CT is shared across label types; check it once per (save_mode, save_format).
            if not os.path.exists(os.path.join(ct_dir, stem + ext)):
                return False
            # Seg is label-type-specific.
            for label_type in label_types:
                suffix  = 'sem' if label_type == 'semantic' else 'inst'
                seg_dir = os.path.join(save_path, f'{seg_base}_{suffix}_seg_{save_mode}')
                if not os.path.exists(os.path.join(seg_dir, stem + ext)):
                    return False
    return True


# --- Entry point ---

def main():
    parser = argparse.ArgumentParser(description='CT nodule dataset processing pipeline.')
    parser.add_argument('--dataset',     required=True, choices=list(DATASET_CONFIGS),
                        help='Dataset to process.')
    parser.add_argument('--mode',        required=True, choices=['nodule', 'roi'],
                        help='Processing mode.')
    parser.add_argument('--sync',          action='store_true',
                        help='Download / update raw data before processing.')
    parser.add_argument('--download-only', action='store_true', dest='download_only',
                        help='Download / update raw data and exit without processing. Implies --sync.')
    parser.add_argument('--reprocess',   action='store_true',
                        help='Re-process series that already have output files.')
    parser.add_argument('--save-mode',   default='3d', dest='save_mode',
                        help='Comma-separated save modes: 3d, 2d, or 3d,2d. Default: 3d.')
    parser.add_argument('--save-format', default='numpy', dest='save_format',
                        help='Comma-separated save formats: numpy, nifti, or numpy,nifti. Default: numpy.')
    parser.add_argument('--no-compress', action='store_false', dest='compress',
                        help='Save uncompressed files (faster writes, larger files). '
                             'Produces .npz for numpy format and .nii for nifti format.')
    parser.add_argument('--workers',     type=int, default=None,
                        help='Number of worker processes. Default: number of available GPUs (or 1 if none).')
    parser.add_argument('--cpu',         action='store_true',
                        help='Force CPU-only execution regardless of GPU availability. Implies --workers 1.')
    parser.add_argument('--view',        action='store_true',
                        help='Open interactive viewer instead of saving NPZ files. Implies --cpu.')
    parser.add_argument('--label-type',  default='semantic', dest='label_type',
                        help='Comma-separated label types: semantic, instance, or semantic,instance. '
                             'Default: semantic. instance is only valid with --mode nodule.')
    args = parser.parse_args()

    VALID_SAVE_MODES   = {'3d', '2d'}
    VALID_SAVE_FORMATS = {'numpy', 'nifti'}
    VALID_LABEL_TYPES  = {'semantic', 'instance'}
    save_modes   = [v.strip() for v in args.save_mode.split(',')]
    save_formats = [v.strip() for v in args.save_format.split(',')]
    label_types  = [v.strip() for v in args.label_type.split(',')]
    invalid_modes        = [m for m in save_modes   if m not in VALID_SAVE_MODES]
    invalid_formats      = [f for f in save_formats if f not in VALID_SAVE_FORMATS]
    invalid_label_types  = [t for t in label_types  if t not in VALID_LABEL_TYPES]
    if invalid_modes:
        parser.error(f'Invalid --save-mode value(s): {invalid_modes}. Choose from {sorted(VALID_SAVE_MODES)}.')
    if invalid_formats:
        parser.error(f'Invalid --save-format value(s): {invalid_formats}. Choose from {sorted(VALID_SAVE_FORMATS)}.')
    if invalid_label_types:
        parser.error(f'Invalid --label-type value(s): {invalid_label_types}. Choose from {sorted(VALID_LABEL_TYPES)}.')
    # Canonical order: semantic writers run before NoduleInstanceSegTransform.
    label_types = sorted(set(label_types), key=lambda t: 0 if t == 'semantic' else 1)

    dataset = args.dataset
    mode    = args.mode

    if 'instance' in label_types and mode != 'nodule':
        parser.error('--label-type instance is only valid with --mode nodule.')

    if dataset not in MODE_CONFIGS.get(mode, {}):
        parser.error(f'Mode "{mode}" is not defined for dataset "{dataset}".')

    target_labels, min_num_segments, seg_base = MODE_CONFIGS[mode][dataset]

    raw_path  = os.path.join(RAW_BASE,  dataset)
    save_path = os.path.join(PROC_BASE, dataset)
    log_dir   = os.path.join(save_path, 'logs')

    # Determine GPU count and effective worker count.
    if args.cpu or args.view:
        n_gpus    = 0
        n_workers = 1
    else:
        n_gpus    = torch.cuda.device_count()
        n_workers = args.workers if args.workers is not None else max(n_gpus, 1)

    # spawn must be set before any multiprocessing objects are created.
    if n_workers > 1:
        mp.set_start_method('spawn', force=True)

    torch.set_grad_enabled(False)

    data_manager = IDCFileSystemDataManager(raw_path, DATASET_CONFIGS[dataset])

    pipeline_kwargs = dict(
        save_path=save_path,
        target_labels=target_labels,
        min_num_segments=min_num_segments,
        seg_base=seg_base,
        save_modes=save_modes,
        save_formats=save_formats,
        compress=args.compress,
        viewer=args.view,
        label_types=label_types,
    )

    if n_workers > 1:
        log_queue = mp.Manager().Queue()
        logger, listener = setup_logging(log_dir, dataset, mode, log_queue=log_queue)
        listener.start()
    else:
        logger, _ = setup_logging(log_dir, dataset, mode)

    logger.info(f'Dataset:   {dataset}  Mode: {mode}  Save modes: {save_modes}  Save formats: {save_formats}  Label types: {label_types}')
    logger.info(f'Raw path:  {raw_path}')
    logger.info(f'Save path: {save_path}')
    logger.info(f'Workers:   {n_workers}  GPUs: {n_gpus}')

    if args.sync or args.download_only:
        logger.info('Syncing raw data...')
        data_manager.sync_data()

    if args.download_only:
        logger.info('Download complete. Exiting (--download-only).')
        return

    # Build task list in main process (D7: already_processed check before workers start).
    all_paths = list(data_manager.get_paths())
    n_total = len(all_paths)

    if not args.reprocess:
        tasks = [
            (ct_path, seg_path_list)
            for ct_path, seg_path_list in all_paths
            if not already_processed(save_path, seg_base, os.path.basename(ct_path),
                                     save_modes, save_formats, label_types, args.compress)
        ]
        n_skipped = n_total - len(tasks)
    else:
        tasks     = list(all_paths)
        n_skipped = 0

    n_tasks = len(tasks)
    n_processed = n_failed = 0

    logger.info(f'Found {n_total} CT series. {n_skipped} already processed, {n_tasks} to run.')

    if n_workers > 1:
        pool_init_args = (n_gpus, log_queue, pipeline_kwargs)
        try:
            with mp.Pool(n_workers, initializer=_worker_init, initargs=pool_init_args) as pool:
                for result in tqdm(pool.imap_unordered(_process_series, tasks), total=n_tasks):
                    if result['ok']:
                        n_processed += 1
                    else:
                        n_failed += 1
        except mp.ProcessError as e:
            logger.error(f'Worker process lost: {e}')
        finally:
            listener.stop()
    else:
        device   = f'cuda:0' if n_gpus > 0 else 'cpu'
        pipeline = build_pipeline(device=device, **pipeline_kwargs)
        for ct_path, seg_path_list in tqdm(tasks, total=n_tasks):
            series_id = os.path.basename(ct_path)
            try:
                pipeline({'ct': ct_path, 'seg_list': seg_path_list}, {})
                n_processed += 1
                logger.debug(f'OK: {series_id}')
            except Exception as e:
                n_failed += 1
                logger.error(f'FAILED: {series_id}')
                logger.error(f'  CT:   {ct_path}')
                logger.error(f'  SEG:  {seg_path_list}')
                logger.error(f'  {type(e).__name__}: {e}')
                logger.debug(traceback.format_exc())

    logger.info(
        f'Done. Total: {n_total}  Processed: {n_processed}  '
        f'Skipped: {n_skipped}  Failed: {n_failed}'
    )


if __name__ == '__main__':
    main()
