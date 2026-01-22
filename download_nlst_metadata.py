from idc_index import index
client = index.IDCClient()

# This fetches the clinical index if you haven't already
client.fetch_index('clinical_index')

# Load the metadata table
df_metadata = client.get_clinical_table('nlst_canc')

# Save table to CSV file
df_metadata.to_csv('./data/nlst_meta.csv')

