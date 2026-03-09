import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    shared = ['age', 'zip3','gender']
    anon_unique = anon_df[~anon_df.duplicated(subset=shared, keep=False)]
    aux_unique = aux_df[~aux_df.duplicated(subset=shared, keep=False)]
    return anon_unique.merge(
        aux_unique,
        on=shared,
        how='inner',
        validate='1:1'
    ).loc[:, ['anon_id', 'name']].rename(columns={'name': 'matched_name'})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return matches_df.shape[0] / anon_df.shape[0]
