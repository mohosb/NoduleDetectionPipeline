from idc_index import index
client = index.IDCClient()

# 1. Ensure clinical index is fresh
client.fetch_index('clinical_index')

# 2. Find the table name for this specific collection
clinical_info = client.clinical_index
nsclc_tables = clinical_info[clinical_info['collection_id'] == 'nsclc_radiomics']
table_name = nsclc_tables['short_table_name'].unique()[0]

# 3. Load the clinical data
df_meta = client.get_clinical_table(table_name)

# 4. Save table into CSV file
df_meta.to_csv('./data/nsclc_meta.csv')

