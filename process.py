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
    python process.py --dataset lidc_idri       --mode nodule --save-mode 2d  # save per-slice
    python process.py --dataset lidc_idri       --mode nodule --workers 4     # use 4 GPU workers
    python process.py --dataset lidc_idri       --mode nodule --cpu           # force CPU single-process
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
)
from ct_data_management.processing.writers import NPZWriter
from ct_data_management.processing.utils import InteractiveViewer, TimePipelinePart


RAW_BASE  = '/mnt/seagate_exp/radiology/data/raw'
PROC_BASE = '/mnt/seagate_exp/radiology/data/processed'

# Per-dataset configuration: (DatasetConfig, raw_subdir, processed_subdir)
DATASET_CONFIGS = {
    'lidc_idri':       LIDC_IDRI_INFO,
    'nsclc_radiomics': NSCLC_RADIOMICS_INFO,
    'nlst_labeled':    NLST_LABELED_INFO,
}

# Per-mode configuration: (target_labels, min_num_segments, seg_save_subdir)
MODE_CONFIGS = {
    'nodule': {
        'lidc_idri':       (['nodule'],           1, 'nodule_seg'),
        'nsclc_radiomics': (['neoplasm'],         1, 'nodule_seg'),
        'nlst_labeled':    (['nodule'],           1, 'nodule_seg'),
    },
    'roi': {
        'nsclc_radiomics': (['lung', 'neoplasm'], 2, 'roi_seg'),
        'nlst_labeled':    (['lung', 'nodule'],   2, 'roi_seg'),
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
                   seg_subdir: str, save_mode: str = '3d', compress: bool = True,
                   device: str = 'cpu'):
    ct_dir  = os.path.join(save_path, f'ct_{save_mode}')
    seg_dir = os.path.join(save_path, f'{seg_subdir}_{save_mode}')

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

    parts.append(NPZWriter(ct_dir, seg_dir, save_mode=save_mode, compress=compress))
    #parts.append(InteractiveViewer())

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
    fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)s  %(message)s'))

    ch = _TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s  %(message)s'))

    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'Logging to {log_file}')

    if log_queue is not None:
        # Workers route records to this queue; listener dispatches them to fh and ch.
        listener = logging.handlers.QueueListener(log_queue, fh, ch, respect_handler_level=True)
        return logger, listener

    return logger, None


# --- Helpers ---

def already_processed(save_path: str, seg_subdir: str, series_id: str, save_mode: str) -> bool:
    ct_dir  = os.path.join(save_path, f'ct_{save_mode}')
    seg_dir = os.path.join(save_path, f'{seg_subdir}_{save_mode}')
    if save_mode == '3d':
        return (os.path.exists(os.path.join(ct_dir,  series_id + '.npz')) and
                os.path.exists(os.path.join(seg_dir, series_id + '.npz')))
    else:  # '2d' — check for the first slice of each series
        return (os.path.exists(os.path.join(ct_dir,  f'{series_id}_0000.npz')) and
                os.path.exists(os.path.join(seg_dir, f'{series_id}_0000.npz')))


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
    parser.add_argument('--save-mode',   default='3d', choices=['3d', '2d'], dest='save_mode',
                        help='3d: one NPZ per volume (default). 2d: one NPZ per axial slice.')
    parser.add_argument('--no-compress', action='store_false', dest='compress',
                        help='Save uncompressed NPZ files (faster writes, larger files).')
    parser.add_argument('--workers',     type=int, default=None,
                        help='Number of worker processes. Default: number of available GPUs (or 1 if none).')
    parser.add_argument('--cpu',         action='store_true',
                        help='Force CPU-only execution regardless of GPU availability. Implies --workers 1.')
    args = parser.parse_args()

    dataset = args.dataset
    mode    = args.mode

    if dataset not in MODE_CONFIGS.get(mode, {}):
        parser.error(f'Mode "{mode}" is not defined for dataset "{dataset}".')

    target_labels, min_num_segments, seg_subdir = MODE_CONFIGS[mode][dataset]

    raw_path  = os.path.join(RAW_BASE,  dataset)
    save_path = os.path.join(PROC_BASE, dataset)
    log_dir   = os.path.join(save_path, 'logs')

    # Determine GPU count and effective worker count.
    if args.cpu:
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
        seg_subdir=seg_subdir,
        save_mode=args.save_mode,
        compress=args.compress,
    )

    if n_workers > 1:
        log_queue = mp.Manager().Queue()
        logger, listener = setup_logging(log_dir, dataset, mode, log_queue=log_queue)
        listener.start()
    else:
        logger, _ = setup_logging(log_dir, dataset, mode)

    logger.info(f'Dataset:   {dataset}  Mode: {mode}  Save mode: {args.save_mode}')
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
            if not already_processed(save_path, seg_subdir, os.path.basename(ct_path), args.save_mode)
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
