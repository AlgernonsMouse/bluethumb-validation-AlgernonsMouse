
"""
test_pipeline.py - Verify pipeline results

Your task: Implement verification tests

Run with: pytest tests/test_pipeline.py -v
"""

import pandas as pd
import pytest
from pathlib import Path
from scipy import stats


def test_matched_pairs_exist():
    """Verify matched pairs file was created"""
    path = Path("data/outputs/matched_pairs.csv")
    assert path.exists(), "matched_pairs.csv not found"


def test_correct_column_names():
    """Verify exact column names match specification (order matters)."""
    expected_columns = [
        'Vol_SiteID', 'Pro_SiteID',
        'Vol_Organization', 'Pro_Organization',
        'Vol_Value', 'Pro_Value',
        'Vol_Units', 'Pro_Units',
        'Vol_DateTime', 'Pro_DateTime',
        'Vol_Lat', 'Vol_Lon', 'Pro_Lat', 'Pro_Lon',
        'Distance_m', 'Time_Diff_hours'
    ]
    df = pd.read_csv('data/outputs/matched_pairs.csv')
    assert list(df.columns) == expected_columns


def test_sample_size():
    """Verify we got exactly 48 matches."""
    df = pd.read_csv('data/outputs/matched_pairs.csv')
    assert len(df) == 48


def test_distance_threshold():
    """Verify all distances <= 100m."""
    df = pd.read_csv('data/outputs/matched_pairs.csv')
    assert pd.to_numeric(df['Distance_m'], errors='coerce').max() <= 100


def test_time_threshold():
    """Verify all time differences <= 48 hours."""
    df = pd.read_csv('data/outputs/matched_pairs.csv')
    assert pd.to_numeric(df['Time_Diff_hours'], errors='coerce').max() <= 48


def test_concentration_filter():
    """Verify professional concentrations > 25 mg/L."""
    df = pd.read_csv('data/outputs/matched_pairs.csv')
    assert pd.to_numeric(df['Pro_Value'], errors='coerce').min() > 25


def test_correlation():
    """Verify R² = 0.839 ± 0.001."""
    df = pd.read_csv('data/outputs/matched_pairs.csv')
    vol_values = pd.to_numeric(df['Vol_Value'], errors='coerce')
    pro_values = pd.to_numeric(df['Pro_Value'], errors='coerce')
    mask = vol_values.notna() & pro_values.notna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(pro_values[mask], vol_values[mask])
    r_squared = r_value ** 2
    assert abs(r_squared - 0.839) < 0.001


def test_slope():
    """Verify slope = 0.712 ± 0.001."""
    df = pd.read_csv('data/outputs/matched_pairs.csv')
    vol_values = pd.to_numeric(df['Vol_Value'], errors='coerce')
    pro_values = pd.to_numeric(df['Pro_Value'], errors='coerce')
    mask = vol_values.notna() & pro_values.notna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(pro_values[mask], vol_values[mask])
    assert abs(slope - 0.712) < 0.001


def test_organizations():
    """Verify correct organizations present."""
    df = pd.read_csv('data/outputs/matched_pairs.csv')
    vol_orgs = set(df['Vol_Organization'].dropna().unique())
    pro_orgs = set(df['Pro_Organization'].dropna().unique())
    assert vol_orgs == {'OKCONCOM_WQX', 'CONSERVATION_COMMISSION'}
    assert pro_orgs == {'OKWRB-STREAMS_WQX', 'O_MTRIBE_WQX'}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

