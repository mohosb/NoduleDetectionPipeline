from dataclasses import dataclass
from idc_index import index
import pandas as pd
import os
import warnings
import pydicom
import shutil
from tqdm import tqdm
from glob import glob


@dataclass(frozen=True)
class DatasetConfig:
    collection_id: str
    ct_query_conditions: str
    seg_query_conditions: str


NSCLC_RADIOMICS_INFO = DatasetConfig(
    collection_id='nsclc_radiomics',
    ct_query_conditions='TRUE',
    seg_query_conditions='SeriesDescription = \'Segmentation\'',
)

LIDC_IDRI_INFO = DatasetConfig(
    collection_id='lidc_idri',
    ct_query_conditions='TRUE',
    seg_query_conditions='SeriesDescription LIKE \'Segmentation of Nodule %\'',
)

NLST_LABELED_INFO = DatasetConfig(
    collection_id='nlst',
    ct_query_conditions='TRUE',
    seg_query_conditions='SeriesDescription LIKE \'AIMI lung and nodule %\'',
)


class DataIntegrityError(Exception):
    pass


class IDCFileSystemDataManager:
    _MANIFEST_FILE = 'manifest.csv'

    def __init__(self, data_path, data_info: DatasetConfig):
        self._data_path = data_path
        self._data_info = data_info

    # --- Public API ---

    def sync_data(self):
        self._download_segmentations()
        self._build_manifest()
        self._download_ct()
        return self

    def get_paths(self):
        manifest = self._load_manifest()

        grouped = (
            manifest
            .groupby('CTSeriesInstanceUID')['SEGSeriesInstanceUID']
            .apply(list)
            .reset_index()
        )

        ct_dir  = os.path.join(self._data_path, 'ct')
        seg_dir = os.path.join(self._data_path, 'seg')

        for row in grouped.itertuples(index=False):
            ct_id     = row.CTSeriesInstanceUID
            seg_ids   = row.SEGSeriesInstanceUID
            ct_path   = os.path.join(ct_dir, ct_id)
            seg_paths = [os.path.join(seg_dir, seg_id) for seg_id in seg_ids]

            if not os.path.isdir(ct_path):
                print(f'Warning: CT directory missing, skipping: {ct_path}')
                continue

            valid_seg_paths = [p for p in seg_paths if os.path.isdir(p)]
            n_missing = len(seg_paths) - len(valid_seg_paths)
            if n_missing:
                print(f'Warning: {n_missing} SEG director(ies) missing for CT {ct_id}.')
            if not valid_seg_paths:
                print(f'No valid SEG directories remain for CT {ct_id}, skipping.')
                continue

            yield ct_path, valid_seg_paths

    # --- Sync stages (each idempotent; can be called independently) ---

    def _download_segmentations(self):
        client = self._make_client()

        print('Querying IDC for SEG series...')
        seg_ids = client.sql_query(f'''
            SELECT SeriesInstanceUID
            FROM   index
            WHERE  collection_id = '{self._data_info.collection_id}'
              AND  Modality = 'SEG'
              AND  {self._data_info.seg_query_conditions}
        ''')['SeriesInstanceUID'].tolist()

        print(f'Downloading {len(seg_ids)} SEG series...')
        seg_dir = os.path.join(self._data_path, 'seg')
        os.makedirs(seg_dir, exist_ok=True)
        client.download_dicom_series(
            seriesInstanceUID=seg_ids,
            downloadDir=seg_dir,
            dirTemplate='%SeriesInstanceUID',
        )

    def _build_manifest(self):
        manifest_path = os.path.join(self._data_path, self._MANIFEST_FILE)

        if os.path.exists(manifest_path):
            existing    = pd.read_csv(manifest_path, header=0)
            known_segs  = set(existing['SEGSeriesInstanceUID'].tolist())
        else:
            existing   = pd.DataFrame(columns=['CTSeriesInstanceUID', 'SEGSeriesInstanceUID'])
            known_segs = set()

        seg_dir = os.path.join(self._data_path, 'seg')
        seg_series_dirs = sorted([
            os.path.join(seg_dir, d)
            for d in os.listdir(seg_dir)
            if os.path.isdir(os.path.join(seg_dir, d))
        ])

        print('Building manifest from SEG DICOM headers...')
        new_pairs = []
        for series_path in tqdm(seg_series_dirs):
            seg_id = os.path.basename(series_path)
            if seg_id in known_segs:
                continue

            dcm_files = glob(os.path.join(series_path, '*.dcm'))
            if not dcm_files:
                print(f'No DICOM files in {series_path}, skipping.')
                continue

            seg_metadata = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            try:
                ct_id = str(seg_metadata.ReferencedSeriesSequence[0].SeriesInstanceUID)
            except (AttributeError, IndexError) as e:
                print(f'Metadata extraction failed for {series_path}: {type(e).__name__}')
                print(f'Removing defective SEG series: {series_path}')
                shutil.rmtree(series_path)
                continue

            new_pairs.append({'CTSeriesInstanceUID': ct_id, 'SEGSeriesInstanceUID': seg_id})

        if new_pairs:
            updated = pd.concat([existing, pd.DataFrame(new_pairs)], ignore_index=True)
            updated.to_csv(manifest_path, index=False)
            print(f'Manifest updated with {len(new_pairs)} new entries → {manifest_path}')
        else:
            print('Manifest is already up to date.')

    def _download_ct(self):
        manifest = self._load_manifest()
        all_ct_ids = manifest['CTSeriesInstanceUID'].unique().tolist()

        ct_dir = os.path.join(self._data_path, 'ct')
        os.makedirs(ct_dir, exist_ok=True)

        missing_ct_ids = [ct for ct in all_ct_ids if not os.path.isdir(os.path.join(ct_dir, ct))]
        if not missing_ct_ids:
            print('All CT series already present on disk.')
            return

        print(f'Downloading {len(missing_ct_ids)}/{len(all_ct_ids)} CT series...')
        client = self._make_client()
        client.download_dicom_series(
            seriesInstanceUID=missing_ct_ids,
            downloadDir=ct_dir,
            dirTemplate='%SeriesInstanceUID',
        )

    # --- Helpers ---

    def _load_manifest(self):
        manifest_path = os.path.join(self._data_path, self._MANIFEST_FILE)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f'Manifest not found at {manifest_path}. Run sync_data() first.'
            )
        return pd.read_csv(manifest_path, header=0)

    @staticmethod
    def _make_client():
        pd.set_option('mode.chained_assignment', None)
        warnings.simplefilter('ignore', FutureWarning)
        warnings.simplefilter('ignore', pd.errors.ChainedAssignmentError)
        return index.IDCClient()
