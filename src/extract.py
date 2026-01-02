"""
extract.py - Download data from EPA Water Quality Portal

Your task: Implement the download function
"""

from pathlib import Path
from datetime import datetime
import zipfile

import pandas as pd
import requests
import yaml


def load_config():
    """Load configuration from YAML file"""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def _format_wqp_date(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value.strftime("%m-%d-%Y")
    s = str(value)
    try:
        # Accept ISO 'YYYY-MM-DD' (used in config.yaml) and convert to WQP 'MM-DD-YYYY'
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%m-%d-%Y")
    except ValueError:
        return s


def _normalize_wqp_provider(value: str) -> str:
    return str(value).strip()


def download_oklahoma_chloride(config):
    """
    Download Oklahoma chloride data from EPA Water Quality Portal

    EPA API Documentation: https://www.waterqualitydata.us/webservices_documentation/

    Args:
        config: Configuration dictionary

    Returns:
        Path to downloaded CSV file
    """

    base_url = "https://www.waterqualitydata.us/data/Result/search"

    data_sources = config["data_sources"]
    date_range = data_sources.get("date_range", {})

    characteristic = data_sources.get("characteristic")
    if isinstance(characteristic, (list, tuple)):
        characteristic_param = ";".join(str(c) for c in characteristic)
    else:
        characteristic_param = str(characteristic)

    providers = data_sources.get("providers", [])

    params = {
        "statecode": data_sources.get("state_code"),
        "characteristicName": characteristic_param,
        "siteType": data_sources.get("site_type"),
        "sampleMedia": data_sources.get("sample_media"),
        "startDateLo": _format_wqp_date(date_range.get("start")),
        "startDateHi": _format_wqp_date(date_range.get("end")),
        "mimeType": "csv",
        "zip": "yes",
        "dataProfile": "resultPhysChem",
    }

    # WQP providers are enumerated, and multiple values should be passed as repeated
    # query params (e.g., providers=NWIS&providers=STORET), not comma-separated.
    if isinstance(providers, (list, tuple)):
        provider_values = [_normalize_wqp_provider(p) for p in providers]
        params = list(params.items()) + [("providers", p) for p in provider_values]
    else:
        params["providers"] = _normalize_wqp_provider(providers)

    output_dir = Path(config["output_paths"]["raw_data"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {config['data_sources']['characteristic']} data from EPA...")
    response = requests.get(base_url, params=params, stream=True, timeout=120)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        warning = response.headers.get("warning")
        if warning:
            print(f"WQP warning header: {warning}")
        if response.text:
            print("WQP response body:")
            print(response.text[:4000])
        raise

    zip_path = output_dir / "oklahoma_data.zip"
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    final_path = output_dir / "oklahoma_chloride.csv"
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        csv_members = [
            m
            for m in members
            if m.lower().endswith(".csv") and ("result" in Path(m).name.lower())
        ]
        if not csv_members:
            raise FileNotFoundError(
                "No result CSV found in the downloaded ZIP. Expected a .csv file with 'result' in the filename."
            )

        result_member = csv_members[0]
        with zf.open(result_member, "r") as src, open(final_path, "wb") as dst:
            while True:
                buf = src.read(1024 * 1024)
                if not buf:
                    break
                dst.write(buf)

    zip_path.unlink()

    df = pd.read_csv(final_path, low_memory=False)
    size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded {len(df):,} records")
    print(f"Saved to {final_path} ({size_mb:.2f} MB)")

    return final_path


def main():
    """Main execution"""
    config = load_config()
    download_oklahoma_chloride(config)
    print("\n✅ Data extraction complete")


if __name__ == "__main__":
    main()
