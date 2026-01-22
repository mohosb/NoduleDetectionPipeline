import pandas as pd
from idc_index import index

# 1. Initialize the IDC Index
client = index.IDCClient()

# 2. Query for CT scans that have NO associated "lung nodule" SEG masks
# Note: Using 'NOT IN' to exclude studies that appeared in our first download
query = '''
SELECT
  SeriesInstanceUID,
  Modality,
  SeriesDescription
FROM
  index
WHERE
  collection_id = 'nlst'
  AND Modality = 'CT'
  AND StudyInstanceUID NOT IN (
    -- This subquery identifies studies that DO have a lung nodule mask
    SELECT StudyInstanceUID
    FROM index
    WHERE collection_id = 'nlst'
      AND Modality = 'SEG'
      AND SeriesDescription LIKE '%lung%'
      AND SeriesDescription LIKE '%nodule%'
  )
'''

# 3. Execute query
df_unlabeled = client.sql_query(query)

# 4. Extract UIDs
series_to_download = df_unlabeled['SeriesInstanceUID'].unique().tolist()

if not series_to_download:
    print('No series found matching those criteria. Please check the spelling in SeriesDescription.')
else:
    print(f'Found {len(series_to_download)} series. Starting download...')

    # 5. Download the series
    client.download_dicom_series(
        seriesInstanceUID=series_to_download,
        downloadDir='./data'
    )


