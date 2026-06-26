# =============================================================================
# 1. Load train.csv & test.csv
# 2. Định nghĩa 5 models (Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM)
# 3. Cross Validation bằng GroupKFold(5) (groups='date') trên tập train để
#    đảm bảo flights cùng ngày không nằm ở cả train fold và val fold.
# 4. Chọn model tốt nhất từ CV (dựa trên RMSE trung bình hoặc R2 trung bình).
# 5. Huấn luyện lại CÁC model 
# 6. Dự đoán trên test, tính Metrics (MAE, RMSE, MAPE, R2, Adjusted R2, Train R2)
# 7. Tính Intercept, Coefficients cho Linear Regression (baseline)
# 8. SHAP analysis cho best model
# 9. Vẽ biểu đồ (Actual vs Predicted, Residuals, Model Comparison, Overfitting, Feature Importance)
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = OUTPUT_DIR / "train.csv"
TEST_CSV  = OUTPUT_DIR / "test.csv"
LOG_FILE  = LOGS_DIR / "02_train_report.txt"

# ---------------------------------------------------------------------------
# Features & Target
# ---------------------------------------------------------------------------
FEATURES = [
    'distance', 'flight_duration', 'speed', 'altitude', 'payload',
    'temperature', 'humidity', 'wind_speed', 'wind_gust', 'wind_dir',
    'pressure', 'cloud_cover', 'relative_wind_angle'
]
TARGET = 'energy_consumption_wh'

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

def calc_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calc_adj_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# =============================================================================
def main():
    log("=" * 70)
    log("02_train_all_models.py — Training & Evaluation (with CV)")
    log("=" * 70)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        log("Error: train.csv or test.csv not found!")
        return

    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    log(f"[1] Loaded train: {len(df_train)} rows, test: {len(df_test)} rows")

    X_train = df_train[FEATURES].values
    y_train = df_train[TARGET].values
    groups_train = df_train['date'].values

    X_test = df_test[FEATURES].values
    y_test = df_test[TARGET].values

    # -----------------------------------------------------------------------
    # Define Models
    # -----------------------------------------------------------------------
    log("\n[2] Defining 5 Models")
    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Decision Tree': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', DecisionTreeRegressor(max_depth=None, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
        ]),
        'LightGBM': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1))
        ])
    }

    # -----------------------------------------------------------------------
    # Cross Validation (GroupKFold)
    # -----------------------------------------------------------------------
    log("\n[3] Cross Validation (GroupKFold = 5) on Train set")
    # Determine max splits based on unique dates
    n_unique_groups = len(np.unique(groups_train))
    n_splits = min(5, n_unique_groups)
    if n_splits < 5:
        log(f"  Warning: Only {n_unique_groups} unique dates. Reducing CV folds to {n_splits}.")

    gkf = GroupKFold(n_splits=n_splits)
    
    cv_results_list = []
    best_cv_score = float('-inf')
    best_model_name = ""

    for name, pipeline in models.items():
        log(f"  Running CV for {name}...")
        cv_scores = cross_validate(
            pipeline, X_train, y_train, groups=groups_train, cv=gkf,
            scoring=('r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'),
            return_train_score=True
        )
        
        mean_r2_val = np.mean(cv_scores['test_r2'])
        mean_mae_val = -np.mean(cv_scores['test_neg_mean_absolute_error'])
        mean_rmse_val = -np.mean(cv_scores['test_neg_root_mean_squared_error'])
        
        cv_results_list.append({
            'Model': name,
            'CV_R2_mean': mean_r2_val,
            'CV_MAE_mean': mean_mae_val,
            'CV_RMSE_mean': mean_rmse_val
        })
        
        log(f"    -> Val R2: {mean_r2_val:.3f} | Val MAE: {mean_mae_val:.3f} Wh")
        
        if mean_r2_val > best_cv_score:
            best_cv_score = mean_r2_val
            best_model_name = name

    cv_df = pd.DataFrame(cv_results_list)
    cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    log(f"\n  => BEST MODEL FROM CV: {best_model_name} (R2={best_cv_score:.3f})")

    # -----------------------------------------------------------------------
    # Train ALL models on FULL train set & Predict on Test set
    # -----------------------------------------------------------------------
    log("\n[4] Training models on FULL train set & Predicting on Test set")
    experiment_results = []
    feature_importances = pd.DataFrame({'Feature': FEATURES})
    
    for name, pipeline in models.items():
        log(f"  Training {name}...")
        pipeline.fit(X_train, y_train)
        
        # Save model
        joblib.Feature_importances = True
        joblib.dump(pipeline, MODELS_DIR / f"{name.replace(' ', '_')}.pkl")
        
        # Train metrics
        y_train_pred = pipeline.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)
        
        # Test metrics
        y_test_pred = pipeline.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = root_mean_squared_error(y_test, y_test_pred)
        mape_test = calc_mape(y_test, y_test_pred)
        adj_r2_test = calc_adj_r2(r2_test, len(y_test), len(FEATURES))
        
        experiment_results.append({
            'Model': name,
            'MAE': mae_test,
            'RMSE': rmse_test,
            'MAPE': mape_test,
            'R2': r2_test,
            'Adj_R2': adj_r2_test,
            'Train_R2': r2_train,
            'R2_Gap (Overfit)': r2_train - r2_test
        })
        
        # Extract feature importances
        model = pipeline.named_steps['regressor']
        if name == 'Linear Regression':
            coeffs = model.coef_
            feature_importances[name] = np.abs(coeffs)
            
            # Save LR coefficients specifically
            lr_df = pd.DataFrame({
                'Feature': FEATURES,
                'Coefficient': coeffs
            }).sort_values('Coefficient', key=abs, ascending=False)
            lr_df.to_csv(RESULTS_DIR / "linear_coefficients.csv", index=False)
            log(f"    -> Linear Regression Intercept: {model.intercept_:.3f}")
        else:
            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = model.feature_importances_
            else:
                feature_importances[name] = 0

    exp_df = pd.DataFrame(experiment_results)
    exp_df.to_csv(RESULTS_DIR / "experiment_log.csv", index=False)
    feature_importances.to_csv(RESULTS_DIR / "feature_importance_all.csv", index=False)
    
    log("\n[5] Test Set Results:")
    log(exp_df.to_string())

    # -----------------------------------------------------------------------
    # SHAP Analysis (Best Model)
    # -----------------------------------------------------------------------
    log(f"\n[6] SHAP Analysis for best model: {best_model_name}")
    best_pipe = models[best_model_name]
    best_regressor = best_pipe.named_steps['regressor']
    
    # We must transform X_test using the scaler before feeding to SHAP explainer
    X_test_scaled = best_pipe.named_steps['scaler'].transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURES)

    try:
        if best_model_name == 'Linear Regression':
            explainer = shap.LinearExplainer(best_regressor, X_test_scaled_df)
            shap_values = explainer.shap_values(X_test_scaled_df)
        else:
            explainer = shap.TreeExplainer(best_regressor)
            shap_values = explainer.shap_values(X_test_scaled_df)
            
        # Save summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_scaled_df, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        log("  SHAP summary plot saved.")
        
        # Save mean absolute SHAP values to CSV
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature': FEATURES,
            'Mean_Abs_SHAP': mean_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        shap_df.to_csv(RESULTS_DIR / "shap_values.csv", index=False)
    except Exception as e:
        log(f"  ⚠ Failed to generate SHAP values: {e}")

    # -----------------------------------------------------------------------
    # Plots (Actual vs Predicted, Residuals, Model Comparison)
    # -----------------------------------------------------------------------
    log("\n[7] Generating Comparison Plots")
    
    # Model Comparison Chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    sns.barplot(data=exp_df, x='Model', y='R2', ax=axes[0], palette='viridis')
    axes[0].set_title('Test R² Score (Higher is better)')
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=exp_df, x='Model', y='MAE', ax=axes[1], palette='viridis')
    axes[1].set_title('Test MAE (Wh) (Lower is better)')
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=exp_df, x='Model', y='R2_Gap (Overfit)', ax=axes[2], palette='coolwarm')
    axes[2].set_title('Train - Test R² Gap (Overfitting)')
    axes[2].axhline(0.1, color='red', linestyle='--', label='Warning Threshold (0.1)')
    axes[2].legend()
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Feature Importance Comparison (normalize across models to compare)
    plt.figure(figsize=(12, 8))
    norm_fi = feature_importances.copy()
    for col in norm_fi.columns:
        if col != 'Feature':
            norm_fi[col] = norm_fi[col] / norm_fi[col].sum()
            
    norm_fi_melt = norm_fi.melt(id_vars='Feature', var_name='Model', value_name='Normalized Importance')
    sns.barplot(data=norm_fi_melt, y='Feature', x='Normalized Importance', hue='Model')
    plt.title('Feature Importance Across Models (Normalized)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Actual vs Predicted & Residuals for best model
    best_pipe = models[best_model_name]
    y_test_pred_best = best_pipe.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, y_test_pred_best, alpha=0.7, color='#4A90D9')
    min_val = min(y_test.min(), y_test_pred_best.min())
    max_val = max(y_test.max(), y_test_pred_best.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0].set_xlabel('Actual Energy (Wh)')
    axes[0].set_ylabel('Predicted Energy (Wh)')
    axes[0].set_title(f'Actual vs Predicted ({best_model_name})')
    
    # Residuals
    residuals = y_test - y_test_pred_best
    axes[1].scatter(y_test_pred_best, residuals, alpha=0.7, color='#E07B39')
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Energy (Wh)')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title(f'Residuals Plot ({best_model_name})')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"actual_vs_predicted_{best_model_name.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    log("  Plots generated and saved.")

    log("\n" + "=" * 70)
    log("PIPELINE COMPLETE")
    log("=" * 70)
    LOG_FILE.write_text("\n".join(log_lines), encoding='utf-8')

if __name__ == "__main__":
    main()
