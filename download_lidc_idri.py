import pandas as pd
from idc_index import index

# 1. Initialize the IDC Index
client = index.IDCClient()

# 2. Corrected SQL query for local DuckDB (idc-index)
# We use 'index' as the table name and standard single quotes
query = '''
SELECT
  SeriesInstanceUID,
  Modality,
  SeriesDescription
FROM
  index
WHERE
  collection_id = 'lidc_idri'
  AND (
    Modality = 'CT' 
    OR Modality = 'SEG'
  )
'''

# 3. Execute query
# idc-index handles the DuckDB connection internally
df = client.sql_query(query)

# 4. Extract UIDs
series_to_download = df['SeriesInstanceUID'].unique().tolist()

if not series_to_download:
    print('No series found matching those criteria. Please check the spelling in SeriesDescription.')
else:
    print(f'Found {len(series_to_download)} series. Starting download...')

    # 5. Download the series
    client.download_dicom_series(
        seriesInstanceUID=series_to_download,
        downloadDir='./data'
    )

