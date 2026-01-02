
"""
visualize.py - Create validation visualizations

Your task: Create a scatter plot comparing volunteer vs professional
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml


def load_config():
    """Load configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)


def create_validation_plot(matches_df, config):
    """Create scatter plot comparing volunteer vs. professional measurements."""

    vol_values = pd.to_numeric(matches_df['Vol_Value'], errors='coerce')
    pro_values = pd.to_numeric(matches_df['Pro_Value'], errors='coerce')
    df = pd.DataFrame({'Vol_Value': vol_values, 'Pro_Value': pro_values}).dropna()
    vol_values = df['Vol_Value'].values
    pro_values = df['Pro_Value'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(pro_values, vol_values)
    r_squared = r_value ** 2

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        pro_values,
        vol_values,
        alpha=0.6,
        color='steelblue',
        s=100,
        edgecolor='white',
        linewidth=0.8,
        label='Matched pairs',
    )

    x_min = float(np.min([pro_values.min(), vol_values.min(), 0]))
    x_max = float(np.max([pro_values.max(), vol_values.max()]))

    x_line = np.linspace(x_min, x_max, 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label='Regression')

    max_val = max(pro_values.max(), vol_values.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='1:1 reference')

    ax.set_xlabel('Professional Chloride (mg/L)', fontsize=12)
    ax.set_ylabel('Volunteer Chloride (mg/L)', fontsize=12)
    ax.set_title('Blue Thumb Virtual Triangulation Validation', fontsize=14, fontweight='bold')

    stats_text = (
        f"N = {len(df)}\n"
        f"R² = {r_squared:.3f}\n"
        f"Slope = {slope:.3f}"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'),
    )

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=True)

    output_path = Path(config['output_paths']['results']) / 'validation_plot.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot: {output_path}")


def create_scatterplot(matches_df, config):
    """Create a simple scatterplot of volunteer vs professional measurements."""

    vol_values = pd.to_numeric(matches_df['Vol_Value'], errors='coerce')
    pro_values = pd.to_numeric(matches_df['Pro_Value'], errors='coerce')
    df = pd.DataFrame({'Vol_Value': vol_values, 'Pro_Value': pro_values}).dropna()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df['Pro_Value'], df['Vol_Value'], alpha=0.7, s=80, color='steelblue')
    ax.set_xlabel('Professional Chloride (mg/L)')
    ax.set_ylabel('Volunteer Chloride (mg/L)')
    ax.set_title('Volunteer vs Professional Chloride (Matched Pairs)')
    ax.grid(True, linestyle='--', alpha=0.4)

    output_path = Path(config['output_paths']['results']) / 'scatterplot.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot: {output_path}")


def main():
    """Create all visualizations"""

    config = load_config()
    matches_path = Path(config['output_paths']['results']) / 'matched_pairs.csv'
    matches_df = pd.read_csv(matches_path, low_memory=False)

    create_validation_plot(matches_df, config)
    create_scatterplot(matches_df, config)

    print("\n✅ Visualization complete")


if __name__ == "__main__":
    main()

