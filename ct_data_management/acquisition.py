import duckdb
from idc_index import index
import pandas as pd
import os
import time
from tqdm import tqdm
from glob import glob
from abc import ABC, abstractmethod, abstractproperty


class DataIntegrityError(Exception):
    pass


class NSCLCRadiomicsDataManager:
    COLLECTION_ID = 'nsclc_radiomics'
    CT_QUERY = f'''
        SELECT
            PatientID,
            StudyInstanceUID,
            SeriesInstanceUID,
            SeriesDescription,
            ImageType
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
            SeriesDescription,
            ImageType
        FROM
            index
        WHERE
            Modality = 'SEG' AND
            SeriesDescription LIKE 'AIMI lung and nodule %';
    '''

    def __init__(self, metadata_path, data_path):
        self._metadata_path = metadata_path
        self._data_path = data_path
        self._metadata_cache = None

    def sync_metadata(self, s3_region='us-east-1'):
        os.makedirs(os.path.dirname(self._metadata_path), exist_ok=True)
        con = duckdb.connect(self._metadata_path)

        con.execute('INSTALL httpfs;')
        con.execute('LOAD httpfs;')

        con.execute('SET threads = 64;')
        con.execute('SET http_keep_alive = true;')
        con.execute(f'SET s3_region=\'{s3_region}\';')

        print(f'Starting ingestion of ALL {self.COLLECTION_ID} metadata columns...')
        print('This pulls from IDC\'s public S3 bucket. It may take a few minutes.')

        start_time = time.time()
        files = [f[0] for f in con.execute('SELECT file FROM glob(\'s3://idc-open-metadata/bigquery_export/idc_current/dicom_all/*.parquet\')').fetchall()]
        for i, file_path in enumerate(tqdm(files)):
            if i == 0: con.execute(f'CREATE OR REPLACE TABLE index AS SELECT * FROM read_parquet(\'{file_path}\')')
            else: con.execute(f'INSERT INTO index SELECT * FROM read_parquet(\'{file_path}\')')
        end_time = time.time()


        #start_time = time.time()
        #query = f'''
        #    EXPLAIN ANALYZE
        #    CREATE OR REPLACE TABLE index AS
        #    SELECT *
        #    FROM
        #        read_parquet('s3://idc-open-metadata/bigquery_export/idc_current/dicom_all/*.parquet')
        #    WHERE
        #        collection_id = '{self.COLLECTION_ID}';
        #'''
        #con.execute(query)
        #end_time = time.time()

        row_count = con.execute('SELECT count(*) FROM index').fetchone()[0]
        col_count = len(con.execute('DESCRIBE index').fetchall())

        con.close()

        print(f'Done! Ingested {row_count} rows and {col_count} columns in {end_time - start_time:.2f} seconds.')
        print(f'Metadata DB saved to: {os.path.abspath(self._metadata_path)}')

        return self

    def sync_data(self):
        if self._metadata_cache is None:
            self._cache_metadata()

        all_series = self._metadata_cache['SeriesInstanceUID_ct'].tolist() + self._metadata_cache['SeriesInstanceUID_seg'].tolist()

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
        if self._metadata_cache is None:
                self._cache_metadata()

        if run_validation: 
            all_db_series = self._metadata_cache['SeriesInstanceUID_ct'].tolist() + self._metadata_cache['SeriesInstanceUID_seg'].tolist()
            all_fs_series = [os.path.basename(p) for p in glob(os.path.join(self._data_path, '*', '*', '*'))]

            if not set(all_db_series) <= set(all_fs_series):
                raise DataIntegrityError('Local files did not match up with metadata.')

        ct_paths = self._metadata_cache.apply(
            lambda r: os.path.join(self._data_path, r['PatientID'], r['StudyInstanceUID'], r['SeriesInstanceUID_ct']),
            axis=1
        )
        seg_paths = self._metadata_cache.apply(
            lambda r: os.path.join(self._data_path, r['PatientID'], r['StudyInstanceUID'], r['SeriesInstanceUID_seg']),
            axis=1
        )
        return zip(ct_paths, seg_paths)

    def _cache_metadata(self):
        if not os.path.exists(self._metadata_path):
            raise FileNotFoundError('Metadata DB could not be found. Check "metadata_path" or run "sync_metadata" first!')
        con = duckdb.connect(self._metadata_path)
        ct_df = con.execute(self.CT_QUERY).df()
        seg_df = con.execute(self.SEG_QUERY).df()
        con.close()

        metadata_df = pd.merge(
            ct_df,
            seg_df,
            on=['PatientID', 'StudyInstanceUID'],
            suffixes=('_ct', '_seg')
        )
        metadata_df['Priority'] = metadata_df['SeriesDescription_seg'].apply(
            lambda description: 1 if 'radiologist' in description.lower() else 2
        )
        metadata_df = metadata_df.sort_values(
            by=['SeriesInstanceUID_ct', 'Priority', 'SeriesDescription_seg'],
            ascending=[True, True, False]  # By description descending so 'radiologist 4' beats 'radiologist 1'
        )
        metadata_df = metadata_df.drop_duplicates(
            subset=['SeriesInstanceUID_ct'],
            keep='first'
        )

        self._metadata_cache = metadata_df

