# ─────────────────────────────────────────────────────────────
# evaluation.py
# Threshold analysis, zone analysis, tier assignment.
# ─────────────────────────────────────────────────────────────

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List

from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve
)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from src.config import (
    OUTPUT_DIR, THRESHOLD_QUARTERLY,
    THRESHOLD_WEEKLY, THRESHOLD_EMERGENCY
)


def run_evaluation(
    xgb_final,
    xgb_initial,
    lr: LogisticRegression,
    X_train, X_test,
    y_train, y_test,
    lr_proba,
    tuned_proba,
    xgb_test_auc: float,
    FEATURE_COLS: List[str],
    logger: logging.Logger
) -> np.ndarray:
    """
    Full evaluation pipeline.
    Returns failure_proba array (used by scoring module).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info('=' * 55)
    logger.info('STEP 5 — MODEL EVALUATION')
    logger.info('=' * 55)

    final_proba   = xgb_final.predict_proba(X_test)[:, 1]
    final_pred    = xgb_final.predict(X_test)
    final_auc     = roc_auc_score(y_test, final_proba)
    failure_proba = xgb_final.predict_proba(X_test)[:, 0]
    total_failures = (y_test == 0).sum()

    # ── ROC Curve + Confusion Matrix + Feature Importance ────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    lr_fpr,   lr_tpr,   _ = roc_curve(y_test, lr_proba)
    init_fpr, init_tpr, _ = roc_curve(y_test, xgb_initial.predict_proba(X_test)[:, 1])
    fin_fpr,  fin_tpr,  _ = roc_curve(y_test, final_proba)
    lr_auc = roc_auc_score(y_test, lr_proba)

    axes[0].plot(lr_fpr,   lr_tpr,   color='#3498db', linewidth=2,
                 label=f'LR (AUC={lr_auc:.3f})')
    axes[0].plot(init_fpr, init_tpr, color='#e67e22', linewidth=2,
                 label=f'XGB initial (AUC={xgb_test_auc:.3f})')
    axes[0].plot(fin_fpr,  fin_tpr,  color='#e74c3c', linewidth=2.5,
                 label=f'XGB final (AUC={final_auc:.3f})')
    axes[0].plot([0,1],[0,1], 'k--', alpha=0.4, label='Random (0.500)')
    axes[0].fill_between(fin_fpr, fin_tpr, alpha=0.1, color='#e74c3c')
    axes[0].set_title('ROC Curve: All 3 Models', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(fontsize=8)

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, final_pred),
        display_labels=['Failure','Success']
    ).plot(ax=axes[1], colorbar=False, cmap='Blues')
    axes[1].set_title('Confusion Matrix — Final XGBoost\nDefault threshold = 0.50',
                      fontsize=12, fontweight='bold')

    fi = pd.DataFrame({
        'Feature'   : FEATURE_COLS,
        'Importance': xgb_final.feature_importances_
    }).sort_values('Importance')
    fi_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fi)))
    axes[2].barh(fi['Feature'], fi['Importance'], color=fi_colors, edgecolor='white')
    axes[2].set_title('Feature Importance — Final XGBoost', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Importance Score')
    for i, v in enumerate(fi['Importance']):
        axes[2].text(v + 0.001, i, str(round(v, 3)), va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_model_evaluation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('✓ Chart saved: 07_model_evaluation.png')

    logger.info('\nClassification Report (default threshold 0.50):')
    report = classification_report(y_test, final_pred, target_names=['Failure','Success'])
    for line in report.split('\n'):
        if line.strip():
            logger.info(f'  {line}')

    zero_imp = fi[fi['Importance'] == 0.0]['Feature'].tolist()
    if zero_imp:
        logger.info(f'Zero importance features: {zero_imp}')
    else:
        logger.info('All features contribute. No dead weight.')

    # ── Threshold Analysis ────────────────────────────────────
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    logger.info('\nThreshold Analysis — Failure Class')
    logger.info('=' * 65)
    logger.info(f'{"Threshold":>10} {"Precision":>10} {"Recall":>10} {"F1":>8} {"Flagged":>10}')
    logger.info('-' * 65)
    for t in thresholds:
        y_pred_t = np.where(failure_proba > t, 0, 1)
        tp = ((y_pred_t==0) & (y_test==0)).sum()
        fp = ((y_pred_t==0) & (y_test==1)).sum()
        fn = ((y_pred_t==1) & (y_test==0)).sum()
        prec    = tp/(tp+fp) if (tp+fp)>0 else 0
        rec     = tp/(tp+fn) if (tp+fn)>0 else 0
        f1      = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        flagged = (y_pred_t == 0).sum()
        marker  = ' <- RECOMMENDED' if t == THRESHOLD_QUARTERLY else (
                  ' <- DEFAULT'     if t == 0.50 else '')
        logger.info(f'{t:>10.2f} {prec:>10.2f} {rec:>10.2f} {f1:>8.2f} {flagged:>10}{marker}')

    # ── Precision-Recall Curve ────────────────────────────────
    precisions, recalls, _ = precision_recall_curve(
        (y_test == 0).astype(int), failure_proba)
    plt.figure(figsize=(10, 5))
    plt.plot(recalls, precisions, color='#e74c3c', linewidth=2.5)
    plt.fill_between(recalls, precisions, alpha=0.1, color='#e74c3c')
    plt.xlabel('Recall (Failure Class)')
    plt.ylabel('Precision (Failure Class)')
    plt.title('Precision-Recall Trade-off\n'
              'Moving right catches more failures but creates more false alarms',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('✓ Chart saved: precision_recall_curve.png')

    # ── Zone Analysis ─────────────────────────────────────────
    buckets      = np.arange(0, 1.0, 0.05)
    bucket_prec  = []
    bucket_cnt   = []
    bucket_real  = []
    for b in buckets:
        mask = (failure_proba >= b) & (failure_proba < b + 0.05)
        if mask.sum() > 0:
            bucket_prec.append((y_test[mask] == 0).mean())
            bucket_cnt.append(mask.sum())
            bucket_real.append((y_test[mask] == 0).sum())
        else:
            bucket_prec.append(0); bucket_cnt.append(0); bucket_real.append(0)

    baseline    = (y_test == 0).mean()
    zone_colors = []
    for b in buckets:
        if b >= THRESHOLD_WEEKLY:      zone_colors.append('#e74c3c')
        elif b >= 0.50:                zone_colors.append('#e67e22')
        elif b >= THRESHOLD_QUARTERLY: zone_colors.append('#f1c40f')
        else:                          zone_colors.append('#95a5a6')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(buckets, bucket_prec, width=0.04,
                color=zone_colors, edgecolor='white', align='edge')
    axes[0].axhline(y=baseline, color='black', linestyle=':',
                    alpha=0.6, label=f'Baseline = {baseline*100:.1f}%')
    axes[0].axhline(y=0.30, color='#2ecc71', linestyle='--',
                    alpha=0.8, label='30% precision line')
    axes[0].axvline(x=THRESHOLD_WEEKLY, color='#e74c3c', linestyle='-',
                    linewidth=2, alpha=0.8, label=f'Action threshold = {THRESHOLD_WEEKLY}')
    axes[0].text(0.12, 0.85, 'DEAD ZONE\n(0.0-0.50)\nbelow baseline',
                 transform=axes[0].transAxes, color='#7f8c8d', fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    axes[0].text(0.88, 0.85, f'SIGNAL ZONE\n({THRESHOLD_WEEKLY}-1.0)\nhigh precision',
                 transform=axes[0].transAxes, color='#e74c3c', fontsize=9, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    axes[0].set_title(f'Precision by Failure Probability Bucket\n'
                      f'Model reliable above {THRESHOLD_WEEKLY}',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Failure Probability')
    axes[0].set_ylabel('Precision')
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 1.05)

    thresholds_sweep  = np.arange(0.10, 1.0, 0.01)
    flagged_counts    = []
    failures_caught   = []
    precisions_sweep  = []
    for t in thresholds_sweep:
        mask    = failure_proba >= t
        flagged = mask.sum()
        caught  = (mask & (y_test == 0).values).sum()
        flagged_counts.append(flagged)
        failures_caught.append(caught / total_failures * 100)
        precisions_sweep.append(caught / flagged if flagged > 0 else 0)

    ax2  = axes[1]
    line1, = ax2.plot(flagged_counts, failures_caught, color='#e74c3c',
                      linewidth=2.5, label='% failures caught')
    ax2.set_xlabel('Number of Trials Reviewed')
    ax2.set_ylabel('% of Real Failures Caught', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2b = ax2.twinx()
    line2, = ax2b.plot(flagged_counts, precisions_sweep, color='#3498db',
                       linewidth=2.5, linestyle='--', label='Precision')
    ax2b.set_ylabel('Precision', color='#3498db')
    ax2b.tick_params(axis='y', labelcolor='#3498db')
    idx_075 = np.argmin(np.abs(thresholds_sweep - THRESHOLD_WEEKLY))
    ax2.axvline(x=flagged_counts[idx_075], color='#2ecc71', linewidth=2)
    ax2.scatter([flagged_counts[idx_075]], [failures_caught[idx_075]],
                color='#e74c3c', s=100, zorder=5)
    ax2b.scatter([flagged_counts[idx_075]], [precisions_sweep[idx_075]],
                 color='#3498db', s=100, zorder=5)
    ax2.set_title(f'Operating Point Analysis\nGreen line = threshold {THRESHOLD_WEEKLY}',
                  fontsize=12, fontweight='bold')
    ax2.legend([line1, line2], ['% failures caught','Precision'], fontsize=9, loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'zone_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('✓ Chart saved: zone_analysis.png')

    logger.info('\nFailure Probability Buckets:')
    logger.info('=' * 65)
    logger.info(f'{"Bucket":<15} {"Trials":>8} {"Real Failures":>14} {"Precision":>10}')
    logger.info('-' * 65)
    for i, b in enumerate(buckets):
        if bucket_cnt[i] > 0:
            flag = ' <- worth investigating' if bucket_prec[i] > 0.30 else ''
            logger.info(f'{b:.2f}-{b+0.05:.2f}  {bucket_cnt[i]:>8}'
                        f'  {bucket_real[i]:>14}  {bucket_prec[i]:>10.2f}{flag}')

    logger.info(f'\nTHRESHOLD DECISION')
    logger.info(f'  Baseline (random flagging) : {baseline*100:.1f}%')
    logger.info(f'  Action threshold           : {THRESHOLD_WEEKLY}')
    logger.info(f'  Above {THRESHOLD_WEEKLY}: precision reliable')
    logger.info(f'  Below {THRESHOLD_WEEKLY}: precision near-random')

    # ── Business Threshold Recommendations ───────────────────
    thresholds_biz = {
        'Quarterly Screening (max recall)'   : THRESHOLD_QUARTERLY,
        'Weekly Operations (recommended)'    : THRESHOLD_WEEKLY,
        'Emergency Escalation (max precision)': THRESHOLD_EMERGENCY,
    }
    logger.info('\nBUSINESS THRESHOLD RECOMMENDATIONS')
    logger.info('=' * 70)
    logger.info(f'{"Use Case":<35} {"Threshold":>10} {"Trials":>8}'
                f' {"Recall":>8} {"Precision":>10}')
    logger.info('-' * 70)
    for use_case, thresh in thresholds_biz.items():
        mask    = failure_proba >= thresh
        flagged = mask.sum()
        caught  = (mask & (y_test == 0).values).sum()
        rec     = caught / total_failures
        prec    = caught / flagged if flagged > 0 else 0
        marker  = ' <- PRIMARY' if thresh == THRESHOLD_WEEKLY else ''
        logger.info(f'{use_case:<35} {thresh:>10.2f} {flagged:>8}'
                    f' {rec:>8.2f} {prec:>10.2f}{marker}')

    # ── Tier Analysis ─────────────────────────────────────────
    def assign_tier(prob):
        if prob >= THRESHOLD_WEEKLY:      return 'CRITICAL'
        elif prob >= 0.50:                return 'HIGH'
        elif prob >= THRESHOLD_QUARTERLY: return 'MODERATE'
        return 'LOW'

    tiers = pd.Series(failure_proba).apply(assign_tier)
    tier_defs = {
        'CRITICAL': f'>= {THRESHOLD_WEEKLY}',
        'HIGH'    : '0.50-0.75',
        'MODERATE': '0.30-0.50',
        'LOW'     : '< 0.30'
    }
    logger.info('\nTIER ANALYSIS')
    logger.info('=' * 75)
    logger.info(f'{"Tier":<12} {"Boundary":>12} {"Flagged":>8}'
                f' {"Real Failures":>14} {"Precision":>10} {"Failures Caught":>16}')
    logger.info('-' * 75)
    for tier in ['CRITICAL','HIGH','MODERATE','LOW']:
        mask      = (tiers == tier).values
        flagged   = mask.sum()
        real_fail = (mask & (y_test == 0).values).sum()
        prec      = real_fail / flagged if flagged > 0 else 0
        pct       = real_fail / total_failures * 100
        logger.info(f'{tier:<12} {tier_defs[tier]:>12} {flagged:>8}'
                    f' {real_fail:>14} {prec:>10.2f} {pct:>15.1f}%')
    logger.info(f'\nTotal actual failures in test set: {total_failures}')

    # ── SHAP Values ───────────────────────────────────────────
    logger.info('\nComputing SHAP values (30-60 seconds)...')
    shap_values = None
    try:
        import shap as shap_lib
        explainer   = shap_lib.TreeExplainer(xgb_final)
        X_test_np   = X_test.values if hasattr(X_test, 'values') else X_test
        sv          = explainer.shap_values(X_test_np)
        shap_values = sv
        mean_abs_shap = pd.DataFrame({
            'feature'   : FEATURE_COLS,
            'importance': np.abs(sv).mean(axis=0)
        }).sort_values('importance', ascending=False)
        logger.info('SHAP Feature Importance (top 5):')
        for _, row in mean_abs_shap.head(5).iterrows():
            logger.info(f'  {row["feature"]:<30}: {row["importance"]:.4f}')
        mean_abs_shap.to_csv(os.path.join(OUTPUT_DIR, 'shap_importance.csv'), index=False)
        logger.info('SHAP importance saved: shap_importance.csv')
    except Exception as e:
        logger.warning(f'SHAP failed: {e}')
        shap_values = None

    logger.info('\n✓ Evaluation complete.')
    return failure_proba, shap_values, FEATURE_COLS
