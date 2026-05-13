'''
Generate a stratified train/test split from a nodule catalog CSV.

The catalog is produced automatically by the processing pipeline (via
NoduleCatalogWriter) and contains one row per nodule with columns:
  patient_id, series_uid, dataset, nodule_id, volume_mm3, entropy

Test set eligibility:
  - lidc_idri        all scans
  - nsclc_radiomics  all scans
  - nlst_radiologist all scans (hard cap: 102)
  - nlst_ai          training only

The split is stratified along dataset, median nodule volume, and mean
nodule entropy (4 quantile bins each).  Sampling is done at the patient
level so no patient appears in both splits.

Usage:
    python generate_split.py nodule_catalog.csv
    python generate_split.py nodule_catalog.csv --test-size 306 --seed 42 --output split.csv
'''

import argparse
import logging
import sys

import numpy as np
import pandas as pd

_logger = logging.getLogger('generate_split')

TEST_ELIGIBLE_DATASETS = {'lidc_idri', 'nsclc_radiomics', 'nlst_radiologist'}
NLST_RADIOLOGIST_CAP   = 102
BALANCE_WARNING_SIZE   = 3 * NLST_RADIOLOGIST_CAP   # 306

N_QUANTILE_BINS = 4


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _aggregate_to_scan(catalog: pd.DataFrame) -> pd.DataFrame:
    return (
        catalog
        .groupby('series_uid', sort=False)
        .agg(
            patient_id    = ('patient_id',  'first'),
            dataset       = ('dataset',     'first'),
            median_volume = ('volume_mm3',  'median'),
            mean_entropy  = ('entropy',     'mean'),
        )
        .reset_index()
    )


def _aggregate_to_patient(per_scan: pd.DataFrame) -> pd.DataFrame:
    return (
        per_scan
        .groupby('patient_id', sort=False)
        .agg(
            dataset       = ('dataset',       'first'),
            median_volume = ('median_volume', 'median'),
            mean_entropy  = ('mean_entropy',  'mean'),
        )
        .reset_index()
    )


def _assign_bins(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    """Quantile-bin a column; fall back to rank bins if duplicate edges exist."""
    try:
        return pd.qcut(df[col], q=n, labels=False, duplicates='drop')
    except ValueError:
        return pd.qcut(df[col].rank(method='first'), q=n, labels=False)


def _stratified_sample(
    patients: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.Index:
    """Sample n patients stratified by (volume_bin, entropy_bin).

    Falls back to proportional random sampling if strata are too sparse.
    """
    if len(patients) <= n:
        return patients.index

    strata   = patients.groupby(['volume_bin', 'entropy_bin'], observed=True)
    total    = len(patients)
    selected = []

    for _, group in strata:
        quota = max(1, round(n * len(group) / total))
        quota = min(quota, len(group))
        selected.extend(rng.choice(group.index, size=quota, replace=False).tolist())

    # Adjust to hit exactly n (rounding may leave us ±a few)
    selected = list(dict.fromkeys(selected))   # deduplicate, preserve order
    if len(selected) < n:
        remaining = patients.index.difference(selected)
        extra = rng.choice(remaining, size=n - len(selected), replace=False)
        selected.extend(extra.tolist())
    elif len(selected) > n:
        selected = rng.choice(selected, size=n, replace=False).tolist()

    return pd.Index(selected)


def generate_split(
    catalog_csv: str,
    test_size: int = BALANCE_WARNING_SIZE,
    seed: int = 42,
    output: str = 'split.csv',
) -> pd.DataFrame:

    catalog = pd.read_csv(catalog_csv)
    required = {'patient_id', 'series_uid', 'dataset', 'volume_mm3', 'entropy'}
    missing  = required - set(catalog.columns)
    if missing:
        raise ValueError(f'Catalog is missing columns: {missing}')

    if test_size > BALANCE_WARNING_SIZE:
        _logger.warning(
            'Requested test size %d exceeds 3 × %d = %d. '
            'NLST radiologist pool is limited to %d scans; '
            'the test set will be underrepresented for that sub-dataset.',
            test_size, NLST_RADIOLOGIST_CAP, BALANCE_WARNING_SIZE, NLST_RADIOLOGIST_CAP,
        )

    per_scan    = _aggregate_to_scan(catalog)
    per_patient = _aggregate_to_patient(per_scan)

    eligible = per_patient[per_patient['dataset'].isin(TEST_ELIGIBLE_DATASETS)].copy()

    # Compute bins across the full eligible pool for globally consistent edges.
    eligible['volume_bin']  = _assign_bins(eligible, 'median_volume', N_QUANTILE_BINS)
    eligible['entropy_bin'] = _assign_bins(eligible, 'mean_entropy',  N_QUANTILE_BINS)

    rng            = np.random.default_rng(seed)
    per_ds_quota   = test_size // 3
    test_patients  = []

    for ds in sorted(TEST_ELIGIBLE_DATASETS):
        pool  = eligible[eligible['dataset'] == ds]
        quota = min(per_ds_quota, NLST_RADIOLOGIST_CAP if ds == 'nlst_radiologist' else len(pool))
        quota = min(quota, len(pool))
        if quota == 0:
            _logger.warning('No eligible patients found for dataset "%s".', ds)
            continue
        selected = _stratified_sample(pool, quota, rng)
        test_patients.extend(selected.tolist())
        _logger.info('Selected %d / %d patients from %s for test.', len(selected), len(pool), ds)

    test_patient_set = set(test_patients)

    # Assign split at the scan level, including all datasets (training pool
    # contains nlst_ai and non-selected scans from the other datasets).
    per_scan['split'] = per_scan['patient_id'].apply(
        lambda pid: 'test' if pid in test_patient_set else 'train'
    )

    result = per_scan[['series_uid', 'patient_id', 'dataset', 'split']]
    result.to_csv(output, index=False)
    _logger.info(
        'Split saved to %s  (test: %d  train: %d)',
        output,
        (result['split'] == 'test').sum(),
        (result['split'] == 'train').sum(),
    )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)-8s %(message)s',
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description='Generate a stratified train/test split from a nodule catalog.',
        epilog='The catalog CSV is produced by running the processing pipeline.',
    )
    parser.add_argument('catalog',    help='Path to nodule_catalog.csv')
    parser.add_argument('--test-size', type=int, default=BALANCE_WARNING_SIZE,
                        help=f'Total number of scans in the test set (default: {BALANCE_WARNING_SIZE})')
    parser.add_argument('--seed',      type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output',    default='split.csv',
                        help='Output CSV path (default: split.csv)')
    args = parser.parse_args()

    generate_split(
        catalog_csv=args.catalog,
        test_size=args.test_size,
        seed=args.seed,
        output=args.output,
    )


if __name__ == '__main__':
    main()
