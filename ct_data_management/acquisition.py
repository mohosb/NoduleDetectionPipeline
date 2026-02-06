import duckdb
from idc_index import index
import pandas as pd
import re
import os
import time
import subprocess
import warnings
from tqdm import tqdm
from glob import glob
from abc import ABC, abstractmethod, abstractproperty
from .utils import SmartTemporaryDirectory


class DataIntegrityError(Exception):
    pass


class NSCLCRadiomicsDataManager:
    COLLECTION_ID = 'nsclc_radiomics'
    CT_QUERY = f'''
        SELECT
            PatientID,
            StudyInstanceUID,
            SeriesInstanceUID,
            SeriesDescription
        FROM
            index
        WHERE
            Modality = 'CT';
    '''
    SEG_QUERY = '''
        SELECT
            PatientID,
            StudyInstanceUID,
            SeriesInstanceUID,
            SeriesDescription
        FROM
            index
        WHERE
            Modality = 'SEG' AND
            SeriesDescription = 'Segmentation';
    '''

    def __init__(self, metadata_path, data_path):
        self._metadata_path = metadata_path
        self._data_path = data_path
        self._metadata_cache = None

    def sync_metadata(self, s3_region='us-east-1'):
        os.makedirs(os.path.dirname(self._metadata_path), exist_ok=True)
        con = duckdb.connect(self._metadata_path)

        env = os.environ.copy()
        env['AWS_REGION'] = s3_region
        result = subprocess.run(
            ['s5cmd', '--no-sign-request', 'du', 's3://idc-open-metadata/bigquery_export/idc_current/dicom_all/*.parquet'],
            env=env,
            capture_output=True,
            text=True,
        )
        num_bytes = int(re.search(r'(\d+) bytes', result.stdout).group(1))
        num_files = int(re.search(r'(\d+) objects', result.stdout).group(1))

        required_space = num_bytes / 1024 ** 3
        print(f'Downloading {num_files} metadata files of {required_space:.2f} GB, from IDC\'s public s3 bucket...')

        with SmartTemporaryDirectory(num_bytes, verbose=True) as temp_dir:
            process = subprocess.Popen(
                ['s5cmd', '--no-sign-request', 'cp', 's3://idc-open-metadata/bigquery_export/idc_current/dicom_all/*.parquet', str(temp_dir)], 
                env=env, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=-1
            )
            try:
                with tqdm(total=num_files, unit='file') as pbar:
                    for line in iter(process.stdout.readline, ''):
                        if 'cp ' in line:
                            pbar.update(1)
            finally:
                process.stdout.close()
                process.wait()
        
            print(f'Ingesting {self.COLLECTION_ID} metadata files...')

            start_time = time.time()
            temp_files = glob(os.path.join(temp_dir, '*.parquet'))
            con.execute('SET preserve_insertion_order = false')  # Saves significant amount of RAM
            con.execute('''
                CREATE OR REPLACE TABLE index AS
                SELECT DISTINCT
                    collection_id,
                    PatientID,
                    StudyInstanceUID,
                    SeriesInstanceUID,
                    Modality,
                    SeriesDescription,
                    ImageType,
                    series_aws_url
                FROM
                    read_parquet(?)
                WHERE
                    collection_id = ?;
            ''', [temp_files, self.COLLECTION_ID])
            end_time = time.time()

        row_count = con.execute('SELECT count(*) FROM index').fetchone()[0]
        col_count = len(con.execute('DESCRIBE index').fetchall())

        con.close()

        print(f'Done! Ingested {row_count} rows and {col_count} columns in {end_time - start_time:.2f} seconds.')
        print(f'Metadata DB saved to: {os.path.abspath(self._metadata_path)}')

        return self

    def sync_data(self):
        with duckdb.connect(self._metadata_path) as con:
            ct_df = con.execute(self.CT_QUERY).df()
            seg_df = con.execute(self.SEG_QUERY).df()

        all_series = ct_df['SeriesInstanceUID'].tolist() + seg_df['SeriesInstanceUID'].tolist()

        pd.set_option('mode.chained_assignment', None)
        warnings.simplefilter('ignore', FutureWarning)
        warnings.simplefilter('ignore', pd.errors.ChainedAssignmentError)

        os.makedirs(self._data_path, exist_ok=True)
        index.IDCClient().download_dicom_series(
            seriesInstanceUID=all_series,
            downloadDir=self._data_path,
            dirTemplate='%PatientID/%StudyInstanceUID/%SeriesInstanceUID'
        )
        return self

    def query_metadata(self, query):
        pass

    def get_paths(self, run_validation=True):
        with duckdb.connect(self._metadata_path) as con:
            ct_df = con.execute(self.CT_QUERY).df()
            seg_df = con.execute(self.SEG_QUERY).df()

        metadata_df = pd.merge(
            ct_df,
            seg_df,
            on=['PatientID', 'StudyInstanceUID'],
            suffixes=('_ct', '_seg')
        )
        metadata_df = metadata_df.drop_duplicates(
            subset=['SeriesInstanceUID_ct'],
            keep='first'
        )

        if run_validation: 
            all_db_series = metadata_df['SeriesInstanceUID_ct'].tolist() + metadata_df['SeriesInstanceUID_seg'].tolist()
            all_fs_series = [os.path.basename(p) for p in glob(os.path.join(self._data_path, '*', '*', '*'))]

            if not set(all_db_series) <= set(all_fs_series):
                raise DataIntegrityError('Local files did not match up with metadata.')

        ct_paths = metadata_df.apply(
            lambda r: os.path.join(self._data_path, r['PatientID'], r['StudyInstanceUID'], r['SeriesInstanceUID_ct']),
            axis=1
        )
        seg_paths = metadata_df.apply(
            lambda r: os.path.join(self._data_path, r['PatientID'], r['StudyInstanceUID'], r['SeriesInstanceUID_seg']),
            axis=1
        )
        return zip(ct_paths, seg_paths)

    
# Querys for NLST:
"""
CT_QUERY = f'''
    SELECT
        PatientID,
        StudyInstanceUID,
        SeriesInstanceUID,
        SeriesDescription
    FROM
        index
    WHERE
        Modality = 'CT' AND
        SeriesDescription LIKE '%STANDARD%' AND
        list_contains(ImageType, 'PRIMARY');
'''
SEG_QUERY = '''
    SELECT
        PatientID,
        StudyInstanceUID,
        SeriesInstanceUID,
        SeriesDescription
    FROM
        index
    WHERE
        Modality = 'SEG' AND
        SeriesDescription LIKE 'AIMI lung and nodule %';
'''
"""

