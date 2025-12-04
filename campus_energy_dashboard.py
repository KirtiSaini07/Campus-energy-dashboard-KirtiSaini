"""
Name: Kirti Saini
Date: 04-121-2025
Title: Campus Energy Visualizer and Analyzer
"""

import os
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------
# Configuration & Logging
# -------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_FILE = OUTPUT_DIR / "ingestion.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# -------------------------
# Helper: Read single CSV robustly
# -------------------------
def read_building_csv(path: Path):
    """
    Read a building CSV robustly, try common column names and parsing.
    Returns DataFrame with columns ['timestamp', 'kwh'] and 'building' metadata (inferred).
    """
    logging.info(f"Reading file: {path}")
    try:
        # Attempt to read CSV; skip bad lines
        df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        logging.error(f"Failed to read {path.name}: {e}")
        return None

    # Normalize column names (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Determine timestamp column
    timestamp_cols = [c for c in df.columns if c in ('timestamp', 'date', 'datetime', 'time')]
    if not timestamp_cols:
        # try to guess datetime-like column
        for c in df.columns:
            if 'date' in c or 'time' in c:
                timestamp_cols = [c]
                break

    if not timestamp_cols:
        logging.error(f"No timestamp column found in {path.name}. Skipping file.")
        return None
    ts_col = timestamp_cols[0]

    # Determine kwh column
    kwh_cols = [c for c in df.columns if c in ('kwh', 'energy_kwh', 'consumption', 'energy', 'usage')]
    if not kwh_cols:
        # fallback to numeric columns not timestamp
        numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        # remove index-like columns
        numeric_candidates = [c for c in numeric_candidates if c != 'index']
        if numeric_candidates:
            kwh_cols = [numeric_candidates[0]]

    if not kwh_cols:
        logging.error(f"No kWh-like column found in {path.name}. Skipping file.")
        return None
    kwh_col = kwh_cols[0]

    # Keep only necessary columns
    df = df[[ts_col, kwh_col]].copy()
    df = df.rename(columns={ts_col: 'timestamp', kwh_col: 'kwh'})

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    n_invalid_dates = df['timestamp'].isna().sum()
    if n_invalid_dates > 0:
        logging.warning(f"{n_invalid_dates} invalid timestamps in {path.name} will be dropped.")
        df = df.dropna(subset=['timestamp'])

    # Coerce kwh to numeric
    df['kwh'] = pd.to_numeric(df['kwh'], errors='coerce')
    n_invalid_kwh = df['kwh'].isna().sum()
    if n_invalid_kwh > 0:
        logging.warning(f"{n_invalid_kwh} non-numeric kwh rows in {path.name} -> will be dropped.")
        df = df.dropna(subset=['kwh'])

    # Add building metadata (from filename if not present)
    building_name = path.stem
    df['building'] = building_name

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    logging.info(f"Loaded {len(df)} rows for building '{building_name}'")
    return df

# -------------------------
# Task 1: Ingest all CSVs into combined DF
# -------------------------
def ingest_all_data(data_dir: Path):
    """
    Reads all csv files from data_dir and returns combined DataFrame.
    Logs missing/corrupt files.
    """
    if not data_dir.exists():
        logging.error(f"Data directory {data_dir} does not exist.")
        return pd.DataFrame(columns=['timestamp', 'kwh', 'building'])

    all_files = list(data_dir.glob("*.csv"))
    if not all_files:
        logging.warning(f"No CSV files found in {data_dir}.")
        return pd.DataFrame(columns=['timestamp', 'kwh', 'building'])

    frames = []
    for f in all_files:
        try:
            df = read_building_csv(f)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            logging.exception(f"Exception while processing {f.name}: {e}")

    if frames:
        df_combined = pd.concat(frames, ignore_index=True)
        # unify types
        df_combined['kwh'] = pd.to_numeric(df_combined['kwh'], errors='coerce')
        df_combined = df_combined.dropna(subset=['timestamp', 'kwh'])
        # set timezone-naive normalization
        df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
        logging.info(f"Combined DataFrame contains {len(df_combined)} rows from {len(frames)} files.")
    else:
        df_combined = pd.DataFrame(columns=['timestamp', 'kwh', 'building'])
        logging.warning("No valid data frames to combine.")

    return df_combined

# -------------------------
# Task 2: Aggregation functions
# -------------------------
def calculate_daily_totals(df):
    """Return daily totals per building and overall."""
    df2 = df.copy()
    df2 = df2.set_index('timestamp')
    # Resample per building
    daily = df2.groupby('building').resample('D')['kwh'].sum().reset_index()
    # Pivot: rows = date, columns = building
    daily_pivot = daily.pivot(index='timestamp', columns='building', values='kwh').fillna(0)
    daily_total = daily_pivot.sum(axis=1).rename('campus_total_kwh')
    return daily_pivot, daily_total

def calculate_weekly_aggregates(df):
    """Return weekly aggregates per building (mean weekly usage)."""
    df2 = df.copy().set_index('timestamp')
    weekly = df2.groupby('building').resample('W')['kwh'].sum().reset_index()
    weekly_pivot = weekly.pivot(index='timestamp', columns='building', values='kwh').fillna(0)
    return weekly_pivot

def building_wise_summary(df):
    """Return summary statistics per building as a DataFrame."""
    summaries = []
    for b, group in df.groupby('building'):
        total = group['kwh'].sum()
        mean = group['kwh'].mean()
        minimum = group['kwh'].min()
        maximum = group['kwh'].max()
        peak_time = group.loc[group['kwh'].idxmax(), 'timestamp']
        summaries.append({
            'building': b,
            'total_kwh': total,
            'mean_kwh': mean,
            'min_kwh': minimum,
            'max_kwh': maximum,
            'peak_timestamp': peak_time
        })
    return pd.DataFrame(summaries).set_index('building')

# -------------------------
# Task 3: OOP Modeling
# -------------------------
class MeterReading:
    def __init__(self, timestamp: pd.Timestamp, kwh: float):
        self.timestamp = pd.to_datetime(timestamp)
        self.kwh = float(kwh)

    def __repr__(self):
        return f"MeterReading({self.timestamp.isoformat()}, {self.kwh})"

class Building:
    def __init__(self, name: str):
        self.name = name
        self.readings = []  # list of MeterReading

    def add_reading(self, mr: MeterReading):
        self.readings.append(mr)

    def add_readings_from_df(self, df: pd.DataFrame):
        """Add multiple readings from a df containing timestamp & kwh"""
        for _, row in df.iterrows():
            self.add_reading(MeterReading(row['timestamp'], row['kwh']))

    def total_consumption(self):
        return sum(r.kwh for r in self.readings)

    def average_consumption(self):
        if not self.readings:
            return 0.0
        return np.mean([r.kwh for r in self.readings])

    def max_consumption(self):
        if not self.readings:
            return 0.0
        max_r = max(self.readings, key=lambda x: x.kwh)
        return max_r.kwh, max_r.timestamp

    def generate_report(self):
        total = self.total_consumption()
        avg = self.average_consumption()
        max_val, max_ts = self.max_consumption()
        report = {
            'building': self.name,
            'total_kwh': total,
            'mean_kwh': avg,
            'max_kwh': max_val,
            'peak_timestamp': max_ts
        }
        return report

class BuildingManager:
    def __init__(self):
        self.buildings = {}  # name -> Building

    def load_from_dataframe(self, df: pd.DataFrame):
        for name, group in df.groupby('building'):
            b = Building(name)
            b.add_readings_from_df(group[['timestamp', 'kwh']])
            self.buildings[name] = b

    def get_building(self, name):
        return self.buildings.get(name)

    def all_reports(self):
        return pd.DataFrame([b.generate_report() for b in self.buildings.values()]).set_index('building')

# -------------------------
# Task 4: Visualization
# -------------------------
def make_dashboard(daily_pivot, weekly_pivot, building_summary, output_path: Path):
    """
    Creates a single PNG (dashboard) with:
      - trend line: campus daily total + building lines
      - bar chart: average weekly usage per building
      - scatter: peak-hour consumption points
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)

    # 1) Trend line: daily totals + per-building lines
    ax = axes[0]
    # Plot building lines
    for col in daily_pivot.columns:
        ax.plot(daily_pivot.index, daily_pivot[col], label=col, alpha=0.6)
    # Campus total
    campus_total = daily_pivot.sum(axis=1)
    ax.plot(daily_pivot.index, campus_total, label='Campus Total', color='black', linewidth=2)
    ax.set_title("Daily Electricity Consumption - Buildings & Campus Total")
    ax.set_xlabel("Date")
    ax.set_ylabel("kWh")
    ax.legend(ncol=2, fontsize='small')

    # 2) Bar chart: average weekly usage per building
    ax = axes[1]
    avg_weekly = weekly_pivot.mean(axis=0).sort_values(ascending=False)
    avg_weekly.plot(kind='bar', ax=ax)
    ax.set_title("Average Weekly Usage per Building")
    ax.set_ylabel("Average kWh per Week")
    ax.set_xlabel("Building")

    # 3) Scatter: peak-hour consumption vs building (we'll plot each building's top N peaks)
    ax = axes[2]
    peak_points = []
    for b in building_summary.index:
        # Extract top 3 consumption readings for building from original data if available
        peak_points.append((b, building_summary.loc[b, 'max_kwh'], building_summary.loc[b, 'peak_timestamp']))
    # Plot scatter: x = building index, y = kwh
    xs = range(len(peak_points))
    ys = [p[1] for p in peak_points]
    labels = [p[0] for p in peak_points]
    ax.scatter(xs, ys, s=80)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Peak kWh reading")
    ax.set_title("Peak Consumption per Building (single highest reading)")

    # Save
    fig.suptitle("Campus Energy-Use Dashboard", fontsize=16)
    fig.savefig(output_path)
    plt.close(fig)
    logging.info(f"Dashboard saved to {output_path}")

# -------------------------
# Task 5: Exports + Summary Text
# -------------------------
def export_results(df_combined, daily_pivot, building_summary, output_dir: Path):
    # Cleaned full dataset
    cleaned_path = output_dir / "cleaned_energy_data.csv"
    df_combined_sorted = df_combined.sort_values(['building', 'timestamp'])
    df_combined_sorted.to_csv(cleaned_path, index=False)
    logging.info(f"Saved cleaned data to {cleaned_path}")

    # Building summary CSV
    summary_csv = output_dir / "building_summary.csv"
    building_summary.to_csv(summary_csv)
    logging.info(f"Saved building summary to {summary_csv}")

    # Summary text (executive summary)
    summary_txt = output_dir / "summary.txt"
    total_campus = df_combined['kwh'].sum()
    highest_building = building_summary['total_kwh'].idxmax() if not building_summary.empty else "N/A"
    highest_value = building_summary['total_kwh'].max() if not building_summary.empty else 0
    # Peak load time (timestamp of campus max)
    idx = df_combined['kwh'].idxmax() if not df_combined.empty else None
    peak_time = df_combined.loc[idx, 'timestamp'] if idx is not None else None

    with open(summary_txt, "w") as f:
        f.write("CAMPUS ENERGY USAGE - EXECUTIVE SUMMARY\n")
        f.write("======================================\n\n")
        f.write(f"Total campus consumption (kWh): {total_campus:.2f}\n")
        f.write(f"Highest-consuming building: {highest_building} (total kWh = {highest_value:.2f})\n")
        f.write(f"Campus peak single-reading time: {peak_time}\n\n")

        f.write("Top-level weekly/daily trends:\n")
        f.write("- Daily totals: see cleaned_energy_data.csv and dashboard image\n")
        f.write("- Weekly averages per building: see building_summary.csv and dashboard\n\n")

        f.write("Notes and anomalies:\n")
        f.write("- Missing rows / corrupt lines were skipped during ingestion (see ingestion.log)\n")
        f.write("- Timeseries aggregated by day/week in local timestamps provided in input\n")
        f.write("- Consider normalizing meter sampling frequency if sensors differ across buildings\n")

    logging.info(f"Saved summary text to {summary_txt}")

# -------------------------
# Main execution
# -------------------------
def main():
    logging.info("=== Campus Energy Dashboard Script Starting ===")
    df_combined = ingest_all_data(DATA_DIR)

    if df_combined.empty:
        logging.error("No input data available. Exiting.")
        return

    # Task 2: Aggregations
    daily_pivot, daily_total = calculate_daily_totals(df_combined)
    weekly_pivot = calculate_weekly_aggregates(df_combined)
    building_summary = building_wise_summary(df_combined)

    # Task 3: OOP manager (example usage)
    manager = BuildingManager()
    manager.load_from_dataframe(df_combined)
    reports_df = manager.all_reports()

    # Merge manager summary into building_summary (to ensure consistent columns)
    # building_summary already contains total, mean, min, max, peak_timestamp
    # ensure column names consistent
    building_summary = building_summary.rename(columns={
        'total_kwh': 'total_kwh',
        'mean_kwh': 'mean_kwh',
        'min_kwh': 'min_kwh',
        'max_kwh': 'max_kwh',
        'peak_timestamp': 'peak_timestamp'
    })

    # Task 4: Visualization
    dashboard_path = OUTPUT_DIR / "dashboard.png"
    make_dashboard(daily_pivot, weekly_pivot, building_summary, dashboard_path)

    # Task 5: Export results
    export_results(df_combined, daily_pivot, building_summary, OUTPUT_DIR)

    logging.info("=== Script completed successfully ===")
    logging.info(f"Outputs written to {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
