# =============================================================================
#   1. Load raw data
#   2. Kiểm tra missing values, duplicates, outliers
#   3. Loại flights không hợp lệ (hover/calibration, missing weather)
#   4. Tính Energy target: Σ(V × |I| × Δt / 3600) per flight
#   5. Tính Distance: Σ haversine(consecutive GPS points) per flight
#   6. Tính Relative Wind Angle: mean|heading - wind_dir| per flight
#   7. Aggregate per-flight: 13 features + 1 target
#   8. Kiểm tra data leakage, multicollinearity, phân bố target
#   9. Lưu flight_dataset_clean.csv
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

for d in [OUTPUT_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "flight_with_weather.csv"
CLEAN_CSV = OUTPUT_DIR / "flight_dataset_clean.csv"
LOG_FILE = LOGS_DIR / "00_clean_report.txt"

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Ngưỡng loại flight không hợp lệ
MIN_DISTANCE_M   = 30.0    # < 30m → hover / calibration
MIN_ENERGY_WH    = 1.0     # < 1 Wh → bất thường (hover test)
MIN_DURATION_S   = 30.0    # < 30s → flight quá ngắn

# Haversine
EARTH_R = 6371000.0  # metres


# =============================================================================
# Load raw data
# =============================================================================
def load_raw(path: Path) -> pd.DataFrame:
    log("=" * 70)
    log("Load raw data")
    log("=" * 70)
    df = pd.read_csv(path)
    log(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} cols")
    log(f"  Unique flights : {df['flight'].nunique()}")
    log(f"  Columns        : {df.columns.tolist()}")
    return df


# =============================================================================
#  — Kiểm tra chất lượng dữ liệu thô
# =============================================================================
def check_raw_quality(df: pd.DataFrame):
    log()
    log("=" * 70)
    log("Raw data quality check")
    log("=" * 70)

    # Missing values
    log("\n[2-A] Missing values per column:")
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    missing = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
    missing = missing[missing["null_count"] > 0]
    if missing.empty:
        log("   Không có missing values trong toàn bộ raw data")
    else:
        log(missing.to_string())

    #  Duplicate rows
    log(f"\n[2-B] Duplicate rows: {df.duplicated().sum():,}")

    #  Battery voltage & current sanity
    log("\n[2-C] Battery sanity:")
    log(f"  Voltage range  : {df['battery_voltage'].min():.3f} – {df['battery_voltage'].max():.3f} V")
    log(f"  Current range  : {df['battery_current'].min():.3f} – {df['battery_current'].max():.3f} A")
    neg_cur = (df['battery_current'] < 0).sum()
    log(f"  Negative current rows (regen/noise): {neg_cur:,}  → sẽ dùng |I|")

    # Weather columns - flights thiếu
    weather_cols = ['temperature_c', 'humidity_pct', 'wind_speed_ms',
                    'wind_gust_ms', 'wind_dir_deg', 'pressure_hpa', 'cloud_cover_pct']
    log("\n[2-D] Flights thiếu weather data:")
    for col in weather_cols:
        bad_flights = df[df[col].isnull()]['flight'].unique()
        if len(bad_flights):
            log(f"  {col:20s}: {len(bad_flights)} flights — {sorted(bad_flights)}")

    #  Speed / payload / altitude distributions
    log("\n[2-E] Operational parameter distributions:")
    log(f"  Speed values   : {sorted(df['speed'].unique())}")
    log(f"  Payload values : {sorted(df['payload'].unique())}")
    log(f"  Altitude values: {sorted(df['altitude'].unique())}")
    log(f"  Routes         : {sorted(df['route'].unique())}")


# =============================================================================
#  Compute per-flight aggregates
# =============================================================================
def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance (metres)."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return EARTH_R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_relative_wind_angle(grp: pd.DataFrame) -> float:
    """
    Relative Wind Angle = mean |flight_heading - wind_direction|
    clamped to [0, 180].
    - flight_heading: tính từ vector GPS consecutive points
    - wind_dir: từ weather data (degrees, meteorological convention)
    """
    lats = grp['position_y'].values
    lons = grp['position_x'].values

    if len(lats) < 2:
        return np.nan

    dx = lons[1:] - lons[:-1]
    dy = lats[1:] - lats[:-1]

    # Loại các điểm tĩnh (drone không di chuyển)
    moving = np.sqrt(dx**2 + dy**2) > 1e-8
    if moving.sum() == 0:
        return np.nan

    headings = np.degrees(np.arctan2(dx, dy)) % 360   # 0°=North, 90°=East
    wind_dir = grp['wind_dir_deg'].values[:-1]

    # Chỉ tính trên rows có weather
    valid = moving & ~np.isnan(wind_dir)
    if valid.sum() == 0:
        return np.nan

    diff = np.abs(headings[valid] - wind_dir[valid]) % 360
    diff = np.where(diff > 180, 360 - diff, diff)
    return float(np.mean(diff))


def aggregate_per_flight(df: pd.DataFrame) -> pd.DataFrame:
    log()
    log("=" * 70)
    log(" Aggregate per-flight features & target")
    log("=" * 70)

    records = []
    for fid, grp in df.groupby('flight', sort=True):
        grp = grp.sort_values('time').reset_index(drop=True)

        # --- Thời gian ---
        t_vals = grp['time'].values
        duration_s = float(t_vals[-1] - t_vals[0]) if len(t_vals) > 1 else 0.0

        # --- Energy target: Σ V × |I| × Δt / 3600 (Wh) ---
        dt = np.diff(t_vals, prepend=t_vals[0])   # Δt per row
        dt[0] = 0.0                                # first row no interval
        power_w = grp['battery_voltage'].values * np.abs(grp['battery_current'].values)
        energy_wh = float(np.sum(power_w * dt / 3600.0))

        # --- Distance: Σ haversine consecutive GPS ---
        lats = grp['position_y'].values
        lons = grp['position_x'].values
        if len(lats) > 1:
            seg = haversine_vec(lats[:-1], lons[:-1], lats[1:], lons[1:])
            distance_m = float(np.nansum(seg))
        else:
            distance_m = 0.0

        # --- Relative Wind Angle ---
        rwa = compute_relative_wind_angle(grp)

        # --- Per-flight constants ---
        speed_ms    = float(grp['speed'].iloc[0])
        payload_g   = float(grp['payload'].iloc[0])
        altitude_m  = float(grp['altitude'].iloc[0])
        route       = grp['route'].iloc[0]
        date        = grp['date'].iloc[0]

        # --- Weather (mean per flight, handle NaN) ---
        temp_c      = grp['temperature_c'].mean()
        humidity    = grp['humidity_pct'].mean()
        wind_spd    = grp['wind_speed_ms'].mean()
        wind_gust   = grp['wind_gust_ms'].mean()
        wind_dir    = grp['wind_dir_deg'].mean()
        pressure    = grp['pressure_hpa'].mean()
        cloud_cover = grp['cloud_cover_pct'].mean()

        records.append({
            'flight_id'             : int(fid),
            'date'                  : date,
            'route'                 : route,
            # --- 13 Features ---
            'distance'              : distance_m,
            'flight_duration'       : duration_s,
            'speed'                 : speed_ms,
            'altitude'              : altitude_m,
            'payload'               : payload_g,
            'temperature'           : temp_c,
            'humidity'              : humidity,
            'wind_speed'            : wind_spd,
            'wind_gust'             : wind_gust,
            'wind_dir'              : wind_dir,
            'pressure'              : pressure,
            'cloud_cover'           : cloud_cover,
            'relative_wind_angle'   : rwa,
            # --- Target ---
            'energy_consumption_wh' : energy_wh,
        })

    flight_df = pd.DataFrame(records)
    log(f"  Aggregated flights: {len(flight_df)}")
    return flight_df


# =============================================================================
#  Loại flights không hợp lệ
# =============================================================================
def filter_invalid_flights(df: pd.DataFrame) -> pd.DataFrame:
    log()
    log("=" * 70)
    log("Filter invalid flights")
    log("=" * 70)
    n_before = len(df)

    reasons = {}

    #  Missing weather data (any weather feature is NaN)
    weather_feats = ['temperature', 'humidity', 'wind_speed', 'wind_gust',
                     'wind_dir', 'pressure', 'cloud_cover', 'relative_wind_angle']
    mask_weather = df[weather_feats].isnull().any(axis=1)
    bad_weather = df.loc[mask_weather, 'flight_id'].tolist()
    reasons['missing_weather'] = bad_weather

    #  Distance < MIN_DISTANCE_M → hover / calibration
    mask_dist = df['distance'] < MIN_DISTANCE_M
    bad_dist = df.loc[mask_dist, 'flight_id'].tolist()
    reasons['distance_too_short'] = bad_dist

    #  Energy < MIN_ENERGY_WH → abnormally low
    mask_energy = df['energy_consumption_wh'] < MIN_ENERGY_WH
    bad_energy = df.loc[mask_energy, 'flight_id'].tolist()
    reasons['energy_too_low'] = bad_energy

    #  Duration < MIN_DURATION_S
    mask_dur = df['flight_duration'] < MIN_DURATION_S
    bad_dur = df.loc[mask_dur, 'flight_id'].tolist()
    reasons['duration_too_short'] = bad_dur

    # Union of all invalid
    invalid_ids = set(bad_weather) | set(bad_dist) | set(bad_energy) | set(bad_dur)

    log(f"\n  Flights flagged per rule:")
    for rule, ids in reasons.items():
        log(f"    {rule:25s}: {len(ids):3d} flights — {sorted(ids)}")

    log(f"\n  Total invalid (union): {len(invalid_ids)} flights — {sorted(invalid_ids)}")
    log(f"\n  Details of removed flights:")
    removed_df = df[df['flight_id'].isin(invalid_ids)][
        ['flight_id', 'date', 'route', 'distance', 'flight_duration',
         'energy_consumption_wh', 'speed', 'payload', 'altitude']
    ].sort_values('flight_id')
    log(removed_df.to_string(index=False))

    df_clean = df[~df['flight_id'].isin(invalid_ids)].reset_index(drop=True)
    log(f"\n  Flights before: {n_before}  →  after: {len(df_clean)}")
    return df_clean


# =============================================================================
# Kiểm tra chất lượng dataset clean
# =============================================================================
def check_clean_quality(df: pd.DataFrame):
    log()
    log("=" * 70)
    log("Clean dataset quality check")
    log("=" * 70)

    FEATURES = [
        'distance', 'flight_duration', 'speed', 'altitude', 'payload',
        'temperature', 'humidity', 'wind_speed', 'wind_gust', 'wind_dir',
        'pressure', 'cloud_cover', 'relative_wind_angle'
    ]
    TARGET = 'energy_consumption_wh'

    # Missing values
    log("\n[5-A] Missing values in clean dataset:")
    null_cnt = df[FEATURES + [TARGET]].isnull().sum()
    if null_cnt.sum() == 0:
        log("  Không có missing values")
    else:
        log(null_cnt[null_cnt > 0].to_string())

    #  Descriptive statistics
    log("\n[5-B] Descriptive statistics:")
    desc = df[FEATURES + [TARGET]].describe().round(3)
    log(desc.to_string())

    # Target distribution
    target = df[TARGET]
    skewness = float(target.skew())
    log(f"\n[5-C] Target (energy_consumption_wh) distribution:")
    log(f"  Count     : {len(target)}")
    log(f"  Mean      : {target.mean():.3f} Wh")
    log(f"  Std       : {target.std():.3f} Wh")
    log(f"  Min       : {target.min():.3f} Wh")
    log(f"  Max       : {target.max():.3f} Wh")
    log(f"  Skewness  : {skewness:.3f}  {'(normal-ish)' if abs(skewness) < 1 else '(skewed — consider log transform)'}")

    #  Data leakage check
    log("\n[5-D] Data leakage check:")
    log("  Features checked against known target-derived columns:")
    leakage_cols = ['battery_voltage', 'battery_current', 'energy_wh',
                    'power_w', 'total_energy']
    for c in leakage_cols:
        exists = c in df.columns
        log(f"    '{c}' in dataset: {exists}  {'POTENTIAL LEAKAGE' if exists else '-> OK'}")

    #  Correlation with target
    log("\n[5-E] Pearson correlation with target (energy_consumption_wh):")
    corr = df[FEATURES].corrwith(df[TARGET]).sort_values(ascending=False).round(3)
    for feat, val in corr.items():
        bar = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        log(f"    {feat:25s}: {val:+.3f}  {sign}{bar}")

    #  Multicollinearity check (correlation between features)
    log("\n[5-F] Feature-feature correlation (>0.85 flagged):")
    feat_corr = df[FEATURES].corr().round(3)
    high_corr_pairs = []
    for i, r in enumerate(FEATURES):
        for j, c in enumerate(FEATURES):
            if i < j and abs(feat_corr.loc[r, c]) > 0.85:
                high_corr_pairs.append((r, c, feat_corr.loc[r, c]))
    if high_corr_pairs:
        for r, c, v in high_corr_pairs:
            log(f"   {r:25s} ↔ {c:25s}: {v:+.3f}")
    else:
        log("  Không có cặp feature nào có tương quan > 0.85")

    #  Overfitting / underfitting risk assessment
    log("\n[5-G] Overfitting / underfitting risk assessment:")
    n = len(df)
    p = len(FEATURES)
    log(f"  Số samples (n)       : {n}")
    log(f"  Số features (p)      : {p}")
    log(f"  n/p ratio            : {n/p:.1f}  {'-> OK (>10)' if n/p >= 10 else 'Low — risk of overfitting for complex models'}")

    #  Distribution by route / speed / payload / altitude
    log("\n[5-H] Distribution by operational parameters:")
    log(f"  Speed distribution   :\n{df['speed'].value_counts().sort_index().to_string()}")
    log(f"\n  Payload distribution :\n{df['payload'].value_counts().sort_index().to_string()}")
    log(f"\n  Altitude distribution:\n{df['altitude'].value_counts().sort_index().to_string()}")
    log(f"\n  Route distribution   :\n{df['route'].value_counts().sort_index().to_string()}")
    log(f"\n  Date distribution    :\n{df['date'].value_counts().sort_index().to_string()}")

    return FEATURES, TARGET


# =============================================================================
# Vẽ biểu đồ EDA
# =============================================================================
def plot_eda(df: pd.DataFrame, features: list, target: str):
    log()
    log("=" * 70)
    log("EDA Plots")
    log("=" * 70)

    # Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[target], bins=25, color='#4A90D9', edgecolor='white', linewidth=0.5)
    axes[0].set_xlabel('Energy Consumption (Wh)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Target Distribution', fontsize=13, fontweight='bold')

    axes[1].hist(np.log1p(df[target]), bins=25, color='#E07B39', edgecolor='white', linewidth=0.5)
    axes[1].set_xlabel('log(Energy Consumption + 1)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Log-transformed Target', fontsize=13, fontweight='bold')

    plt.tight_layout()
    out = FIGURES_DIR / "00_target_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {out}")

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(13, 10))
    cols = features + [target]
    corr_mat = df[cols].corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(
        corr_mat, mask=mask, annot=True, fmt=".2f", cmap='RdYlGn',
        vmin=-1, vmax=1, linewidths=0.5, ax=ax,
        annot_kws={"size": 7}
    )
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = FIGURES_DIR / "00_correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {out}")

    # Feature vs Target scatter (top 6 by |corr|)
    corr_vals = df[features].corrwith(df[target]).abs().sort_values(ascending=False)
    top6 = corr_vals.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for i, feat in enumerate(top6):
        axes[i].scatter(df[feat], df[target], alpha=0.6, s=30, color='#4A90D9', edgecolors='none')
        # Trend line
        z = np.polyfit(df[feat].fillna(df[feat].mean()), df[target], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes[i].plot(x_line, p_line(x_line), 'r--', linewidth=1.5, label=f'r={corr_vals[feat]:.2f}')
        axes[i].set_xlabel(feat, fontsize=10)
        axes[i].set_ylabel('Energy (Wh)', fontsize=10)
        axes[i].set_title(f'{feat} vs Energy', fontsize=11, fontweight='bold')
        axes[i].legend(fontsize=9)
    plt.suptitle('Top-6 Features vs Energy Consumption', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "00_feature_scatter_top6.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {out}")

    # Energy by speed & payload (box plots)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df.boxplot(column=target, by='speed', ax=axes[0], grid=False,
               boxprops=dict(color='#4A90D9'),
               medianprops=dict(color='#E07B39', linewidth=2))
    axes[0].set_title('Energy by Speed (m/s)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Speed (m/s)', fontsize=10)
    axes[0].set_ylabel('Energy (Wh)', fontsize=10)
    plt.suptitle('')

    df.boxplot(column=target, by='payload', ax=axes[1], grid=False,
               boxprops=dict(color='#4A90D9'),
               medianprops=dict(color='#E07B39', linewidth=2))
    axes[1].set_title('Energy by Payload (g)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Payload (g)', fontsize=10)
    axes[1].set_ylabel('Energy (Wh)', fontsize=10)
    plt.suptitle('')

    plt.tight_layout()
    out = FIGURES_DIR / "00_energy_by_operational.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {out}")

    log("   EDA plots saved to figures/")


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 70)
    log("00_check_clean_data.py — Research 1 Data Preparation")
    log("=" * 70)

    # Load
    df_raw = load_raw(RAW_CSV)

    # Quality check trên raw
    check_raw_quality(df_raw)

    # Aggregate per-flight
    flight_df = aggregate_per_flight(df_raw)

    # Filter invalid flights
    flight_clean = filter_invalid_flights(flight_df)

    # Quality check trên clean
    features, target = check_clean_quality(flight_clean)

    # EDA plots
    plot_eda(flight_clean, features, target)

    # Save
    flight_clean.to_csv(CLEAN_CSV, index=False)
    log()
    log("=" * 70)
    log(f"  Clean dataset saved to: {CLEAN_CSV}")
    log(f"  Shape: {flight_clean.shape[0]} flights × {flight_clean.shape[1]} columns")
    log(f"  Columns: {flight_clean.columns.tolist()}")
    log("=" * 70)

    # Write log
    LOG_FILE.write_text("\n".join(log_lines), encoding='utf-8')
    log(f"  Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
