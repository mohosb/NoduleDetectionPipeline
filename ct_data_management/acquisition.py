import duckdb
from idc_index import index
import pandas as pd
import re
import os
import time
import subprocess
import warnings
import pydicom
import shutil
from tqdm import tqdm
from glob import glob
from abc import ABC, abstractmethod, abstractproperty
from .utils import SmartTemporaryDirectory


class NSCLC_RADIOMICS_INFO:
    COLLECTION_ID = 'nsclc_radiomics'
    CT_QUERY_CONDITIONS = '''
        Modality = 'CT';
    '''
    SEG_QUERY_CONDITIONS = ''' 
        Modality = 'SEG' AND
        SeriesDescription = 'Segmentation';
    '''


class NLST_LABELED_INFO:
    COLLECTION_ID = 'nlst'
    CT_QUERY_CONDITIONS = '''
        TRUE
    '''
    SEG_QUERY_CONDITIONS = '''
        SeriesDescription LIKE 'AIMI lung and nodule %'
    '''


class DataIntegrityError(Exception):
    pass


class IDCFileSystemDataManager:
    def __init__(self, data_path, data_info):
        self._data_path = data_path
        self._data_info = data_info

    def sync_data(self):
        #TODO: Add proper support for unlabeled data!
        client = index.IDCClient()

        pd.set_option('mode.chained_assignment', None)
        warnings.simplefilter('ignore', FutureWarning)
        warnings.simplefilter('ignore', pd.errors.ChainedAssignmentError)

        print('Downloading SEG DICOM files...')

        seg_ids = client.sql_query(f'''
            SELECT
                SeriesInstanceUID
            FROM
                index
            WHERE
                collection_id = '{self._data_info.COLLECTION_ID}' AND
                Modality = 'SEG' AND
                {self._data_info.SEG_QUERY_CONDITIONS}
        ''')['SeriesInstanceUID'].tolist()

        seg_dir = os.path.join(self._data_path, 'seg')
        os.makedirs(os.path.dirname(seg_dir), exist_ok=True)
        client.download_dicom_series(
            seriesInstanceUID=seg_ids,
            downloadDir=seg_dir,
            dirTemplate='%SeriesInstanceUID'
        )

        print('Digesting data...')

        ct_ids = []
        for series_path in tqdm(glob(os.path.join(seg_dir, '*'))):
            seg_file = glob(os.path.join(series_path, '*.dcm'))[0]
            seg_metadata = pydicom.dcmread(seg_file, stop_before_pixels=True)

            try:
                ct_ids.append(str(seg_metadata.ReferencedSeriesSequence[0].SeriesInstanceUID))
            except (AttributeError, IndexError) as e:
                print(f'Metadata extraction failed due to: {type(e).__name__}')
    
                if os.path.exists(series_path):
                    print(f'Removing defective SEG DICOM series: {series_path}')
                    shutil.rmtree(series_path)

        print('Downloading CT DICOM files...')

        ct_dir = os.path.join(self._data_path, 'ct')
        os.makedirs(os.path.dirname(ct_dir), exist_ok=True)
        client.download_dicom_series(
            seriesInstanceUID=ct_ids,
            downloadDir=ct_dir,
            dirTemplate='%SeriesInstanceUID'
        )
 
        metadata_path = os.path.join(self._data_path, 'metadata.csv')
        print('Saving metadata to' + metadata_path)

        pd.DataFrame(
            {'CTSeriesInstanceUID': ct_ids, 'SEGSeriesInstanceUID': seg_ids}
        ).to_csv(metadata_path, index=False, header=True)

        return self

    def get_paths(self):
        metadata_path = os.path.join(self._data_path, 'metadata.csv')
        metadata = pd.read_csv(metadata_path, header=0)

        for row in metadata.itertuples(index=False):
            yield os.path.join(self._data_path, 'ct', row[0]), os.path.join(self._data_path, 'seg', row[1])

