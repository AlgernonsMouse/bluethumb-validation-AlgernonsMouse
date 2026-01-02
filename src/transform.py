
"""
transform.py - Clean and filter EPA data

Your task: Implement all the cleaning functions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def _resolve_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns found: {candidates}")


def load_config():
    """Load configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_raw_data(config):
    """
    Load raw EPA data

    Hints:
    - Path is in config['output_paths']['raw_data']
    - Filename is 'oklahoma_chloride.csv'
    - Use low_memory=False to avoid dtype warnings
    """
    raw_dir = Path(config['output_paths']['raw_data'])
    filepath = raw_dir / 'oklahoma_chloride.csv'
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded raw data: {len(df):,} records")
    return df


def filter_chloride(df):
    """
    Filter for chloride measurements only

    Hints:
    - Column name: 'CharacteristicName'
    - Keep only rows where CharacteristicName == 'Chloride'
    - Use .copy() to avoid SettingWithCopyWarning

    Expected output: ~50,000 records
    """
    characteristic_col = _resolve_column(df, ['CharacteristicName'])
    out = df[df[characteristic_col] == 'Chloride'].copy()
    print(f"After chloride filter: {len(out):,} records")
    return out


def clean_coordinates(df, config):
    """
    Remove invalid coordinates

    Note: We rely on the state code filter during extraction rather than
    strict lat/lon bounds. This preserves edge sites near state borders
    that may have valid data.

    Hints:
    - Columns: 'LatitudeMeasure', 'LongitudeMeasure'
    - Remove null values with .notna()
    - DO NOT filter by Oklahoma bounds - state code already handled this

    Expected output: Similar to input (most coordinates are valid)
    """
    lat_col = _resolve_column(df, ['LatitudeMeasure', 'ActivityLocation/LatitudeMeasure'])
    lon_col = _resolve_column(df, ['LongitudeMeasure', 'ActivityLocation/LongitudeMeasure'])

    out = df[df[lat_col].notna() & df[lon_col].notna()].copy()
    print(f"After coordinate cleaning: {len(out):,} records")
    return out


def clean_concentrations(df):
    """
    Filter for valid concentration values

    Hints:
    - Column: 'ResultMeasureValue'
    - Remove null values
    - Check for 'ResultDetectionConditionText' column
      * If it exists, remove rows where it's not null (these are "Not Detected")
    - Remove negative values
    - DO NOT remove high values (>1000 mg/L) - these are scientifically valid
      in cases of industrial discharge or saltwater intrusion

    Expected output: ~45,000 records
    """
    result_value_col = _resolve_column(df, ['ResultMeasureValue'])

    out = df.copy()
    out[result_value_col] = pd.to_numeric(out[result_value_col], errors='coerce')
    out = out[out[result_value_col].notna()].copy()

    if 'ResultDetectionConditionText' in out.columns:
        out = out[out['ResultDetectionConditionText'].isna()].copy()

    out = out[out[result_value_col] >= 0].copy()

    print(f"After concentration cleaning: {len(out):,} records")
    return out


def parse_dates(df):
    """
    Convert dates to datetime objects

    Hints:
    - Column: 'ActivityStartDate'
    - Use pd.to_datetime() with errors='coerce'
    - Remove rows where date parsing failed (null after conversion)

    Expected output: ~44,000 records (very few fail)
    """
    activity_date_col = _resolve_column(df, ['ActivityStartDate'])

    out = df.copy()
    out[activity_date_col] = pd.to_datetime(out[activity_date_col], errors='coerce')
    out = out[out[activity_date_col].notna()].copy()
    print(f"After date parsing: {len(out):,} records")
    return out


def separate_volunteer_professional(df, config):
    """
    Separate volunteer and professional measurements

    Hints:
    - Column: 'OrganizationIdentifier'
    - Volunteer orgs from config['organizations']['volunteer']
    - Professional orgs from config['organizations']['professional']
    - Use .isin() to filter
    - Apply >25 mg/L filter to PROFESSIONAL data only
      * This is in config['matching_parameters']['min_concentration_mg_l']

    Expected output:
    - Volunteer: ~15,819 records
    - Professional: ~21,975 records (after concentration filter)
    """
    org_col = _resolve_column(df, ['OrganizationIdentifier'])
    result_value_col = _resolve_column(df, ['ResultMeasureValue'])

    volunteer_orgs = config['organizations']['volunteer']
    professional_orgs = config['organizations']['professional']
    min_prof = float(config['matching_parameters']['min_concentration_mg_l'])

    volunteer_df = df[df[org_col].isin(volunteer_orgs)].copy()
    professional_df = df[df[org_col].isin(professional_orgs)].copy()

    professional_df = professional_df[professional_df[result_value_col] > min_prof].copy()

    print(f"Volunteer records: {len(volunteer_df):,}")
    print(f"Professional records: {len(professional_df):,}")
    return volunteer_df, professional_df


def save_processed_data(volunteer_df, professional_df, config):
    """
    Save processed datasets

    Hints:
    - Output directory from config
    - Filenames: 'volunteer_chloride.csv', 'professional_chloride.csv'
    - Use .to_csv() with index=False
    """
    output_dir = Path(config['output_paths']['processed_data'])
    output_dir.mkdir(parents=True, exist_ok=True)

    volunteer_path = output_dir / 'volunteer_chloride.csv'
    professional_path = output_dir / 'professional_chloride.csv'

    volunteer_df.to_csv(volunteer_path, index=False)
    professional_df.to_csv(professional_path, index=False)

    print(f"Saved volunteer data: {volunteer_path}")
    print(f"Saved professional data: {professional_path}")


def main():
    """Main data cleaning pipeline"""
    config = load_config()

    df = load_raw_data(config)
    df = filter_chloride(df)
    df = clean_coordinates(df, config)
    df = clean_concentrations(df)
    df = parse_dates(df)
    volunteer_df, professional_df = separate_volunteer_professional(df, config)
    save_processed_data(volunteer_df, professional_df, config)

    print("\n✅ Data transformation complete")


if __name__ == "__main__":
    main()

