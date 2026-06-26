# =============================================================================
#   1. Load flight_dataset_clean.csv
#   2. Định nghĩa 13 input features + 1 target
#   3. GroupShuffleSplit theo 'date' (tránh data leakage — flights cùng ngày
#      có cùng điều kiện thời tiết) → ~75% train / ~25% test 
#      -> lý do chọn 75/25 vì dataset có 4 ngày bay, chia 3/1 ngày sẽ gần 75/25, việc chia 2/2 ngày sẽ dẫn đến train/test không cân bằng về số lượng flights
#      -> loại bỏ missing chỉ còn 194 flight -> train=145, test=49 -> cân bằng hơn về số lượng flights giữa train/test
#   4. In thống kê split
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR   = BASE_DIR / "logs"

CLEAN_CSV  = OUTPUT_DIR / "flight_dataset_clean.csv"
TRAIN_CSV  = OUTPUT_DIR / "train.csv"
TEST_CSV   = OUTPUT_DIR / "test.csv"
LOG_FILE   = LOGS_DIR / "01_split_report.txt"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature / Target definitions  (13 features, 1 target)
# ---------------------------------------------------------------------------
FEATURES = [
    # -- Flight --
    'distance',             # Tổng quãng đường bay (m)
    'flight_duration',      # Thời gian bay (s)
    'speed',                # Tốc độ lệnh (m/s)
    'altitude',             # Độ cao hành trình (m)
    # -- Operational --
    'payload',              # Tải trọng (g)
    # -- Weather --
    'temperature',          # Nhiệt độ (°C)
    'humidity',             # Độ ẩm (%)
    'wind_speed',           # Tốc độ gió (m/s)
    'wind_gust',            # Gió giật (m/s)
    'wind_dir',             # Hướng gió (°)
    'pressure',             # Áp suất khí quyển (hPa)
    'cloud_cover',          # Độ che phủ mây (%)
    # -- Engineered --
    'relative_wind_angle',  # Góc gió tương đối (°) — headwind vs tailwind
]

TARGET = 'energy_consumption_wh'

# ---------------------------------------------------------------------------
log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


# =============================================================================
def main():
    log("=" * 70)
    log("01_train_test_split.py — Feature Selection & Train/Test Split")
    log("=" * 70)

    # -----------------------------------------------------------------------
    #  Load clean dataset
    # -----------------------------------------------------------------------
    log("\n[1] Load clean dataset")
    df = pd.read_csv(CLEAN_CSV)
    log(f"  Loaded: {len(df)} flights × {df.shape[1]} columns")
    log(f"  Columns: {df.columns.tolist()}")

    # Validate required columns exist
    required = FEATURES + [TARGET, 'date', 'flight_id']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in clean dataset: {missing_cols}")
    log("  All required columns present")

    # -----------------------------------------------------------------------
    # Validate features (no NaN allowed at this point)
    # -----------------------------------------------------------------------
    log("\n[2] Feature validation")
    nan_check = df[FEATURES + [TARGET]].isnull().sum()
    bad = nan_check[nan_check > 0]
    if not bad.empty:
        log(f"  Found NaN in features/target:")
        log(bad.to_string())
        # Drop rows with NaN
        n_before = len(df)
        df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
        log(f"  Dropped {n_before - len(df)} rows with NaN → {len(df)} remain")
    else:
        log("  No NaN found in features or target")

    # -----------------------------------------------------------------------
    # Summary statistics of features
    # -----------------------------------------------------------------------
    log("\n[3] Feature + Target statistics")
    stats = df[FEATURES + [TARGET]].describe().round(3)
    log(stats.to_string())

    # -----------------------------------------------------------------------
    # GroupShuffleSplit by 'date'
    #    Flights cùng ngày bay → cùng thời tiết → phải vào cùng split
    # -----------------------------------------------------------------------
    log("\n[4] GroupShuffleSplit by date (train≈75%, test≈25%)")

    dates     = df['date'].values
    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)
    log(f"  Unique dates: {n_dates}  — {unique_dates}")

    X = df[FEATURES].values
    y = df[TARGET].values
    groups = dates   # group key for split

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    train_dates = sorted(df_train['date'].unique())
    test_dates  = sorted(df_test['date'].unique())

    log(f"\n  Train: {len(df_train)} flights ({len(df_train)/len(df)*100:.1f}%)")
    log(f"    Dates in train: {train_dates}")
    log(f"  Test : {len(df_test)} flights ({len(df_test)/len(df)*100:.1f}%)")
    log(f"    Dates in test : {test_dates}")

    # Verify no date overlap
    overlap = set(train_dates) & set(test_dates)
    if overlap:
        log(f"  DATE OVERLAP DETECTED: {overlap}")
    else:
        log("  No date overlap between train and test — data leakage prevented")

    # -----------------------------------------------------------------------
    # Distribution comparison: train vs test
    # -----------------------------------------------------------------------
    log("\n[5] Distribution comparison: Train vs Test")
    log(f"  {'Feature':<25} {'Train mean':>12} {'Test mean':>12} {'Train std':>12} {'Test std':>12}")
    log("  " + "-" * 75)
    for feat in FEATURES + [TARGET]:
        tr_mean = df_train[feat].mean()
        te_mean = df_test[feat].mean()
        tr_std  = df_train[feat].std()
        te_std  = df_test[feat].std()
        log(f"  {feat:<25} {tr_mean:>12.3f} {te_mean:>12.3f} {tr_std:>12.3f} {te_std:>12.3f}")

    # -----------------------------------------------------------------------
    # Energy distribution check
    # -----------------------------------------------------------------------
    log(f"\n[6] Energy distribution check:")
    log(f"  Full  — mean={df[TARGET].mean():.3f} Wh, std={df[TARGET].std():.3f}, "
        f"min={df[TARGET].min():.3f}, max={df[TARGET].max():.3f}")
    log(f"  Train — mean={df_train[TARGET].mean():.3f} Wh, std={df_train[TARGET].std():.3f}, "
        f"min={df_train[TARGET].min():.3f}, max={df_train[TARGET].max():.3f}")
    log(f"  Test  — mean={df_test[TARGET].mean():.3f} Wh, std={df_test[TARGET].std():.3f}, "
        f"min={df_test[TARGET].min():.3f}, max={df_test[TARGET].max():.3f}")

    # -----------------------------------------------------------------------
    # Plot split summary
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Train vs Test energy distributions
    axes[0].hist(df_train[TARGET], bins=20, alpha=0.7, color='#4A90D9',
                 label=f'Train (n={len(df_train)})', edgecolor='white')
    axes[0].hist(df_test[TARGET],  bins=20, alpha=0.7, color='#E07B39',
                 label=f'Test (n={len(df_test)})', edgecolor='white')
    axes[0].set_xlabel('Energy Consumption (Wh)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Train / Test Energy Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()

    # Date allocation
    date_assignment = []
    for d in unique_dates:
        if d in train_dates:
            date_assignment.append(('Train', d, len(df_train[df_train['date'] == d])))
        else:
            date_assignment.append(('Test', d, len(df_test[df_test['date'] == d])))

    split_df = pd.DataFrame(date_assignment, columns=['Split', 'Date', 'Flights'])
    colors = ['#4A90D9' if s == 'Train' else '#E07B39' for s in split_df['Split']]
    bars = axes[1].barh(split_df['Date'], split_df['Flights'], color=colors)
    axes[1].set_xlabel('Number of Flights', fontsize=11)
    axes[1].set_title('Flights per Date (Blue=Train, Orange=Test)', fontsize=11, fontweight='bold')
    axes[1].invert_yaxis()
    for bar, n in zip(bars, split_df['Flights']):
        axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(n), va='center', fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / "01_train_test_split.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"\n  Split plot saved: {out}")

    # -----------------------------------------------------------------------
    # Save train.csv and test.csv
    # -----------------------------------------------------------------------
    save_cols = ['flight_id', 'date', 'route'] + FEATURES + [TARGET]
    df_train[save_cols].to_csv(TRAIN_CSV, index=False)
    df_test[save_cols].to_csv(TEST_CSV, index=False)

    log()
    log("=" * 70)
    log(f"train.csv saved: {len(df_train)} flights — {TRAIN_CSV}")
    log(f"test.csv  saved: {len(df_test)} flights  — {TEST_CSV}")
    log(f"  Train dates: {train_dates}")
    log(f"  Test  dates: {test_dates}")
    log("=" * 70)

    LOG_FILE.write_text("\n".join(log_lines), encoding='utf-8')
    log(f"Log saved: {LOG_FILE}")


if __name__ == "__main__":
    main()
