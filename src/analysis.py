
"""
analysis.py - Virtual triangulation matching algorithm

Your task: Implement the Haversine distance and matching algorithm

This is the core of the project - take your time!
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import yaml


def load_config():
    """Load configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def _resolve_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns found: {candidates}")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points on Earth."""

    R = 6_371_000

    lat1_r, lon1_r, lat2_r, lon2_r = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (np.sin(dlat / 2) ** 2) + np.cos(lat1_r) * np.cos(lat2_r) * (np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _bbox_close_enough(vol_lat, vol_lon, pro_lat, pro_lon, max_distance_m):
    lat_tol = max_distance_m / 111_320
    cos_lat = np.cos(np.radians(vol_lat))
    if cos_lat <= 0:
        lon_tol = 180
    else:
        lon_tol = max_distance_m / (111_320 * cos_lat)

    return (abs(pro_lat - vol_lat) <= lat_tol) and (abs(pro_lon - vol_lon) <= lon_tol)


def find_matches(volunteer_df, professional_df, config):
    """Find volunteer-professional measurement pairs via spatial-temporal thresholds."""

    max_distance_m = config['matching_parameters']['max_distance_meters']
    max_time_hours = config['matching_parameters']['max_time_hours']
    match_strategy = config['matching_parameters']['match_strategy']

    vol_lat_col = _resolve_column(volunteer_df, ['LatitudeMeasure', 'ActivityLocation/LatitudeMeasure'])
    vol_lon_col = _resolve_column(volunteer_df, ['LongitudeMeasure', 'ActivityLocation/LongitudeMeasure'])
    pro_lat_col = _resolve_column(professional_df, ['LatitudeMeasure', 'ActivityLocation/LatitudeMeasure'])
    pro_lon_col = _resolve_column(professional_df, ['LongitudeMeasure', 'ActivityLocation/LongitudeMeasure'])

    datetime_col = _resolve_column(volunteer_df, ['ActivityStartDate'])
    pro_datetime_col = _resolve_column(professional_df, ['ActivityStartDate'])

    value_col = _resolve_column(volunteer_df, ['ResultMeasureValue'])
    pro_value_col = _resolve_column(professional_df, ['ResultMeasureValue'])

    site_col = _resolve_column(volunteer_df, ['MonitoringLocationIdentifier'])
    pro_site_col = _resolve_column(professional_df, ['MonitoringLocationIdentifier'])

    org_col = _resolve_column(volunteer_df, ['OrganizationIdentifier'])
    pro_org_col = _resolve_column(professional_df, ['OrganizationIdentifier'])

    matches = []

    print(f"\nMatching volunteer measurements to professional...")
    print(f"  Volunteer measurements: {len(volunteer_df):,}")
    print(f"  Professional measurements: {len(professional_df):,}")
    print(f"  Max distance: {max_distance_m}m")
    print(f"  Max time: {max_time_hours}hrs")
    print(f"  Strategy: {match_strategy}")
    print(f"\nThis will take a while. Progress bar below:")

    vol_units_col = 'ResultMeasure/MeasureUnitCode' if 'ResultMeasure/MeasureUnitCode' in volunteer_df.columns else None
    pro_units_col = 'ResultMeasure/MeasureUnitCode' if 'ResultMeasure/MeasureUnitCode' in professional_df.columns else None

    vol_lat = volunteer_df[vol_lat_col].to_numpy()
    vol_lon = volunteer_df[vol_lon_col].to_numpy()
    vol_time = pd.to_datetime(volunteer_df[datetime_col], errors='coerce').to_numpy(dtype='datetime64[ns]')
    vol_value = volunteer_df[value_col].to_numpy()
    vol_site = volunteer_df[site_col].to_numpy()
    vol_org = volunteer_df[org_col].to_numpy()
    vol_units = volunteer_df[vol_units_col].to_numpy() if vol_units_col else None

    professional_sorted = professional_df.sort_values(pro_datetime_col).reset_index(drop=True)
    pro_lat = professional_sorted[pro_lat_col].to_numpy()
    pro_lon = professional_sorted[pro_lon_col].to_numpy()
    pro_time = pd.to_datetime(professional_sorted[pro_datetime_col], errors='coerce').to_numpy(dtype='datetime64[ns]')
    pro_value = professional_sorted[pro_value_col].to_numpy()
    pro_site = professional_sorted[pro_site_col].to_numpy()
    pro_org = professional_sorted[pro_org_col].to_numpy()
    pro_units = professional_sorted[pro_units_col].to_numpy() if pro_units_col else None

    time_window = np.timedelta64(int(max_time_hours), 'h')

    for i in tqdm(range(len(volunteer_df)), total=len(volunteer_df)):
        if np.isnat(vol_time[i]):
            continue

        vlat = vol_lat[i]
        vlon = vol_lon[i]
        vdt = vol_time[i]
        vval = vol_value[i]
        vsite = vol_site[i]
        vorg = vol_org[i]
        vunits = vol_units[i] if vol_units is not None else np.nan

        start = vdt - time_window
        end = vdt + time_window
        left = int(np.searchsorted(pro_time, start, side='left'))
        right = int(np.searchsorted(pro_time, end, side='right'))

        candidates = []
        for j in range(left, right):
            if np.isnat(pro_time[j]):
                continue
            plat = pro_lat[j]
            plon = pro_lon[j]
            if not _bbox_close_enough(vlat, vlon, plat, plon, max_distance_m):
                continue

            time_diff = abs((pro_time[j] - vdt) / np.timedelta64(1, 'h'))
            if time_diff > max_time_hours:
                continue

            distance = haversine_distance(vlat, vlon, plat, plon)
            if distance > max_distance_m:
                continue

            candidates.append({
                'distance': float(distance),
                'time_diff': float(time_diff),
                'pro_value': pro_value[j],
                'pro_units': (pro_units[j] if pro_units is not None else np.nan),
                'pro_org': pro_org[j],
                'pro_site_id': pro_site[j],
                'pro_datetime': pd.Timestamp(pro_time[j]).to_pydatetime(),
                'pro_lat': plat,
                'pro_lon': plon,
            })

        if match_strategy == 'all':
            for candidate in candidates:
                matches.append({
                    'Vol_SiteID': vsite,
                    'Pro_SiteID': candidate['pro_site_id'],
                    'Vol_Organization': vorg,
                    'Pro_Organization': candidate['pro_org'],
                    'Vol_Value': vval,
                    'Pro_Value': candidate['pro_value'],
                    'Vol_Units': vunits,
                    'Pro_Units': candidate['pro_units'],
                    'Vol_DateTime': pd.Timestamp(vdt).to_pydatetime(),
                    'Pro_DateTime': candidate['pro_datetime'],
                    'Vol_Lat': vlat,
                    'Vol_Lon': vlon,
                    'Pro_Lat': candidate['pro_lat'],
                    'Pro_Lon': candidate['pro_lon'],
                    'Distance_m': candidate['distance'],
                    'Time_Diff_hours': candidate['time_diff'],
                })
        else:
            if len(candidates) > 0:
                candidates.sort(key=lambda x: x['distance'])
                best_match = candidates[0]
                matches.append({
                    'Vol_SiteID': vsite,
                    'Pro_SiteID': best_match['pro_site_id'],
                    'Vol_Organization': vorg,
                    'Pro_Organization': best_match['pro_org'],
                    'Vol_Value': vval,
                    'Pro_Value': best_match['pro_value'],
                    'Vol_Units': vunits,
                    'Pro_Units': best_match['pro_units'],
                    'Vol_DateTime': pd.Timestamp(vdt).to_pydatetime(),
                    'Pro_DateTime': best_match['pro_datetime'],
                    'Vol_Lat': vlat,
                    'Vol_Lon': vlon,
                    'Pro_Lat': best_match['pro_lat'],
                    'Pro_Lon': best_match['pro_lon'],
                    'Distance_m': best_match['distance'],
                    'Time_Diff_hours': best_match['time_diff'],
                })

    return pd.DataFrame(matches)


def calculate_statistics(matches_df):
    """Calculate correlation and regression statistics."""

    df = matches_df[['Vol_Value', 'Pro_Value']].copy()
    df['Vol_Value'] = pd.to_numeric(df['Vol_Value'], errors='coerce')
    df['Pro_Value'] = pd.to_numeric(df['Pro_Value'], errors='coerce')
    df = df.dropna(subset=['Vol_Value', 'Pro_Value'])

    vol_values = df['Vol_Value'].values
    pro_values = df['Pro_Value'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(pro_values, vol_values)
    r_squared = r_value ** 2

    return {
        'n': int(len(df)),
        'r_squared': float(r_squared),
        'slope': float(slope),
        'intercept': float(intercept),
        'p_value': float(p_value),
    }


def save_results(matches_df, stats_dict, config):
    """Save matched pairs and summary statistics."""

    output_dir = Path(config['output_paths']['results'])
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output_dir / 'matched_pairs.csv'
    stats_path = output_dir / 'summary_statistics.txt'

    matches_df.to_csv(pairs_path, index=False)

    with open(stats_path, 'w') as f:
        f.write(f"N = {stats_dict['n']}\n")
        f.write(f"R^2 = {stats_dict['r_squared']}\n")
        f.write(f"Slope = {stats_dict['slope']}\n")
        f.write(f"Intercept = {stats_dict['intercept']}\n")
        f.write(f"p-value = {stats_dict['p_value']}\n")

    print(f"Saved matched pairs: {pairs_path}")
    print(f"Saved statistics: {stats_path}")


def main():
    """Run virtual triangulation analysis"""

    config = load_config()

    processed_dir = Path(config['output_paths']['processed_data'])
    volunteer_path = processed_dir / 'volunteer_chloride.csv'
    professional_path = processed_dir / 'professional_chloride.csv'

    volunteer_df = pd.read_csv(volunteer_path, low_memory=False)
    professional_df = pd.read_csv(professional_path, low_memory=False)

    # Parse dates (saved as strings in CSV)
    volunteer_df['ActivityStartDate'] = pd.to_datetime(volunteer_df['ActivityStartDate'], errors='coerce')
    professional_df['ActivityStartDate'] = pd.to_datetime(professional_df['ActivityStartDate'], errors='coerce')
    volunteer_df = volunteer_df[volunteer_df['ActivityStartDate'].notna()].copy()
    professional_df = professional_df[professional_df['ActivityStartDate'].notna()].copy()

    matches_df = find_matches(volunteer_df, professional_df, config)
    print(f"\nMatches found: {len(matches_df)}")

    if len(matches_df) == 0:
        print("No matches found; skipping statistics and save.")
        return

    stats_dict = calculate_statistics(matches_df)
    print(f"Regression stats: N={stats_dict['n']} R^2={stats_dict['r_squared']}")

    save_results(matches_df, stats_dict, config)
    print("\n✅ Virtual triangulation analysis complete")


if __name__ == "__main__":
    main()

