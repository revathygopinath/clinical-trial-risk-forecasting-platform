"""
main.py
───────
ClinicalEdge — master pipeline script.
Runs all 8 steps in order. Matches notebook output exactly.

Usage:
    python main.py

Output:
    - Terminal: full progress log with AUC, thresholds, results
    - outputs/: all PNG charts and CSV files
    - logs/: timestamped log file
"""

import os
import sys
import time

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger             import get_logger
from src.config             import OUTPUT_DIR, LOG_DIR, DEPLOY_SPONSOR
from src.data_loader        import load_and_validate
from src.eda                import run_eda
from src.feature_engineering import build_features
from src.model_training     import train_model
from src.evaluation         import run_evaluation
from src.scoring            import score_all_trials, score_active_trials
from src.simulation         import run_simulation


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = get_logger('ClinicalEdge')
    start  = time.time()

    logger.info('╔══════════════════════════════════════════════════════╗')
    logger.info('║           CLINICALEDGE PIPELINE                     ║')
    logger.info('║   Clinical Trial Risk Scoring + Pipeline Forecast   ║')
    logger.info('╚══════════════════════════════════════════════════════╝')
    logger.info(f'Deploy sponsor: {DEPLOY_SPONSOR}')
    logger.info(f'Output folder : {OUTPUT_DIR}')
    logger.info('')

    # ── Step 1: Load Data ─────────────────────────────────────
    df_raw = load_and_validate(logger)

    # ── Step 2: EDA ───────────────────────────────────────────
    run_eda(df_raw, logger)

    # ── Step 3: Feature Engineering ───────────────────────────
    filtered_data, ml_data, FEATURE_COLS, overall_rate, ta_feature_cols = \
        build_features(df_raw, logger)

    # ── Step 4: Model Training ────────────────────────────────
    (xgb_final, X_train, X_test, y_train, y_test,
     mean_auc, std_auc, lr, scaler, lr_proba,
     xgb_initial, tuned_proba, spw) = train_model(filtered_data, FEATURE_COLS, logger)

    # Compute xgb_test_auc for evaluation module
    from sklearn.metrics import roc_auc_score
    xgb_test_auc = roc_auc_score(y_test, xgb_initial.predict_proba(X_test)[:, 1])

    # ── Step 5: Evaluation ────────────────────────────────────
    failure_proba, shap_values, _ = run_evaluation(
        xgb_final, xgb_initial, lr,
        X_train, X_test, y_train, y_test,
        lr_proba, tuned_proba, xgb_test_auc,
        FEATURE_COLS, logger
    )

    # ── Step 6: Score All Finished Trials ─────────────────────
    df_scored = score_all_trials(ml_data, xgb_final, FEATURE_COLS, logger)

    # ── Step 7: Score Active Trials ───────────────────────────
    active_ready = score_active_trials(
        df_raw, xgb_final, FEATURE_COLS, ta_feature_cols,
        overall_rate, ml_data, logger
    )

    # ── Step 8: Monte Carlo Simulation ────────────────────────
    run_simulation(active_ready, logger)

    # ── Summary ───────────────────────────────────────────────
    elapsed = round(time.time() - start, 1)
    logger.info('')
    logger.info('╔══════════════════════════════════════════════════════╗')
    logger.info('║                PIPELINE COMPLETE                    ║')
    logger.info('╚══════════════════════════════════════════════════════╝')
    logger.info(f'Time elapsed          : {elapsed}s')
    logger.info(f'Rolling window AUC    : {mean_auc:.4f} ± {std_auc:.4f}')
    logger.info(f'Final model test AUC  : {xgb_test_auc:.4f}')
    logger.info(f'Active trials scored  : {len(active_ready):,}  ({DEPLOY_SPONSOR})')
    logger.info(f'Outputs saved to      : {OUTPUT_DIR}')
    logger.info('')
    logger.info('FILES GENERATED:')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        logger.info(f'  {f:<50} {size/1024:.1f} KB')
    logger.info('')
    logger.info('Next step: streamlit run dashboard/app.py')


if __name__ == '__main__':
    main()
