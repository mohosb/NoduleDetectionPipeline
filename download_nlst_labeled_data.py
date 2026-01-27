import os
import sys
import pandas as pd
from idc_index import index


CT_QUERY = '''
SELECT
  PatientID,
  StudyInstanceUID,
  SeriesInstanceUID,
  SeriesDescription
FROM
  index
WHERE
  collection_id = 'nlst' AND
  Modality = 'CT' AND
  SeriesDescription LIKE '%STANDARD%'
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
  collection_id = 'nlst' AND
  Modality = 'SEG' AND
  SeriesDescription LIKE 'AIMI lung and nodule %'
'''


if __name__ == '__main__':
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
    else:
        save_path = '../data/raw/'
    os.makedirs(save_path, exist_ok=True)


    client = index.IDCClient()

    ct_df = client.sql_query(CT_QUERY)
    seg_df = client.sql_query(SEG_QUERY)


    df = pd.merge(
        ct_df,
        seg_df,
        on=['PatientID', 'StudyInstanceUID'],
        suffixes=('Ct', 'Seg')
    )

    df['Priority'] = df['SeriesDescriptionSeg'].apply(
        lambda description: 1 if 'radiologist' in description.lower() else 2
    )

    df = df.sort_values(
        by=['PatientID', 'StudyInstanceUID', 'SeriesInstanceUIDCt', 'Priority', 'SeriesDescriptionSeg'],
        ascending=[True, True, True, True, False]  # By description descending so 'radiologist 4' beats 'radiologist 1'
    )
    df = df.drop_duplicates(
        subset=['SeriesInstanceUIDCt'],
        keep='first'
    )

    client.download_dicom_series(
        seriesInstanceUID=df['SeriesInstanceUIDCt'].tolist() + df['SeriesInstanceUIDSeg'].tolist(),
        downloadDir=save_path
    )

