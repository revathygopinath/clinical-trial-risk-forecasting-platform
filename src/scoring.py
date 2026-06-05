# ─────────────────────────────────────────────────────────────
# scoring.py
# Score all finished trials + active sponsor trials.
# ─────────────────────────────────────────────────────────────

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple

from src.config import (
    OUTPUT_DIR, BIG_PHARMA, ACTIVE_STATUSES,
    DEPLOY_SPONSOR, THRESHOLD_WEEKLY,
    THRESHOLD_QUARTERLY, THRESHOLD_EMERGENCY, PHASE_MAP
)


def _map_ta(condition: str) -> str:
    c = str(condition).lower()
    if any(w in c for w in ['cancer','neoplasm','tumor','carcinoma',
                             'leukemia','lymphoma','melanoma','sarcoma','myeloma']):
        return 'Oncology'
    elif any(w in c for w in ['diabetes','insulin','glucose']):
        return 'Metabolic'
    elif any(w in c for w in ['heart','cardiac','hypertension',
                               'cardiovascular','cholesterol']):
        return 'Cardiovascular'
    elif any(w in c for w in ['asthma','pulmonary','copd','respiratory']):
        return 'Respiratory'
    elif any(w in c for w in ['arthritis','psoriasis','lupus',
                               'rheumatoid','crohn','colitis']):
        return 'Immunology'
    elif any(w in c for w in ['hiv','hepatitis','infection',
                               'influenza','virus','bacterial']):
        return 'Infectious Disease'
    elif any(w in c for w in ['alzheimer','parkinson','schizophrenia',
                               'depression','anxiety','neurolog']):
        return 'Neurology'
    return 'Other'


def score_all_trials(
    ml_data: pd.DataFrame,
    xgb_final,
    FEATURE_COLS: List[str],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Score all finished historical trials.
    Returns df_scored with risk_score and risk_label columns.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info('=' * 55)
    logger.info('STEP 6 — RISK SCORING (ALL FINISHED TRIALS)')
    logger.info('=' * 55)

    df_scored = ml_data.dropna(subset=FEATURE_COLS).copy()
    df_scored['success_probability'] = xgb_final.predict_proba(
        df_scored[FEATURE_COLS])[:, 1]
    df_scored['risk_score'] = 1 - df_scored['success_probability']
    df_scored['risk_label'] = pd.cut(
        df_scored['risk_score'],
        bins=[0, 0.30, 0.50, 0.75, 1.0],
        labels=['Low Risk','Moderate Risk','High Risk','Critical Risk']
    )

    rc = df_scored['risk_label'].value_counts()
    logger.info(f'Risk scores generated for {len(df_scored):,} trials')
    for label in ['Critical Risk','High Risk','Moderate Risk','Low Risk']:
        if label in rc.index:
            count = rc[label]
            logger.info(f'  {label:<20} {count:>6,}  ({count/len(df_scored)*100:.1f}%)')
    logger.info(f'Average success probability: {df_scored["success_probability"].mean()*100:.1f}%')
    logger.info(f'Critical Risk (> {THRESHOLD_WEEKLY})    : {(df_scored["risk_label"]=="Critical Risk").sum():,}')

    # Risk scores chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df_scored['success_probability'], bins=50,
                 color='#3498db', edgecolor='white', alpha=0.85)
    axes[0].axvline(0.25, color='orange', linestyle='--', linewidth=2,
                    label=f'Critical boundary (risk > {THRESHOLD_WEEKLY})')
    axes[0].set_title('Distribution of Success Probabilities',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Success Probability')
    axes[0].legend()

    rc_colors = {'Low Risk':'#2ecc71','Moderate Risk':'#f39c12',
                 'High Risk':'#e67e22','Critical Risk':'#e74c3c'}
    axes[1].bar(rc.index, rc.values,
                color=[rc_colors.get(r,'#95a5a6') for r in rc.index],
                edgecolor='white')
    axes[1].set_title('Portfolio Risk Profile\n(Boundaries from zone analysis)',
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Trials')
    for i, v in enumerate(rc.values):
        axes[1].text(i, v + 20, str(v), ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '09_risk_scores.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('✓ Chart saved: 09_risk_scores.png')

    out_cols = ['NCT','Sponsor','Phase','Status','Condition',
                'Start_Year','Enrollment','therapeutic_area',
                'success','success_probability','risk_score','risk_label']
    out_cols = [c for c in out_cols if c in df_scored.columns]
    csv_path = os.path.join(OUTPUT_DIR, 'risk_scores.csv')
    df_scored[out_cols].to_csv(csv_path, index=False)
    logger.info(f'✓ CSV saved: risk_scores.csv')

    return df_scored


def score_active_trials(
    df_raw: pd.DataFrame,
    xgb_final,
    FEATURE_COLS: List[str],
    ta_feature_cols: List[str],
    overall_rate: float,
    ml_data: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Score active trials for DEPLOY_SPONSOR.
    Mirrors notebook active trial scoring cell exactly.
    Returns active_ready DataFrame.
    """
    logger.info('=' * 55)
    logger.info(f'STEP 7 — ACTIVE TRIAL SCORING ({DEPLOY_SPONSOR})')
    logger.info('=' * 55)
    logger.info(f'Model trained on ALL sponsors for maximum signal.')
    logger.info(f'Deployment scoped to: {DEPLOY_SPONSOR}')

    # Filter to active trials for deploy sponsor only
    df_active = df_raw[
        df_raw['Status'].isin(ACTIVE_STATUSES) &
        (df_raw['Sponsor'] == DEPLOY_SPONSOR)
    ].copy()

    logger.info(f'\nActive trials found : {len(df_active):,}')
    logger.info('Status breakdown:')
    for s, v in df_active['Status'].value_counts().items():
        logger.info(f'  {s:<35} {v:>5,}')

    if len(df_active) == 0:
        logger.warning(f'No active trials found for {DEPLOY_SPONSOR}.')
        logger.warning(f'Available sponsors: {df_raw["Sponsor"].value_counts().head(5).index.tolist()}')
        return pd.DataFrame()

    # Feature engineering for active trials
    df_active = df_active.sort_values('Start_Year').reset_index(drop=True)
    df_active['phase_num'] = df_active['Phase'].map(PHASE_MAP).fillna(0).astype(int)

    p99 = df_raw['Enrollment'].quantile(0.99)
    df_active['enrollment_log'] = np.log1p(df_active['Enrollment'].clip(upper=p99))
    df_active['enrollment_bucket'] = pd.cut(
        df_active['Enrollment'],
        bins=[-1, 0, 50, 200, 500, 1000, float('inf')],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    df_active['therapeutic_area'] = df_active['Condition'].apply(_map_ta)
    ta_active = pd.get_dummies(df_active['therapeutic_area'], prefix='ta', drop_first=True)
    for col in ta_feature_cols:
        if col not in ta_active.columns:
            ta_active[col] = 0
    ta_active = ta_active[ta_feature_cols]
    df_active = pd.concat([df_active, ta_active], axis=1)

    df_active['is_big_pharma_raw'] = df_active['Sponsor'].isin(BIG_PHARMA).astype(int)

    # Use sponsor rates learned from training data
    sponsor_rates = ml_data.groupby('Sponsor')['success'].mean().to_dict()
    df_active['sponsor_success_rate'] = (
        df_active['Sponsor'].map(sponsor_rates).fillna(overall_rate)
    )

    df_active['start_decade'] = df_active['Start_Year'].apply(
        lambda y: 0 if y < 1990 else (1 if y < 2000 else (2 if y < 2010 else 3))
    )
    df_active['phase_x_enrollment'] = df_active['phase_num'] * df_active['enrollment_log']
    df_active['phase_x_bigpharma']  = df_active['phase_num'] * df_active['is_big_pharma_raw']

    active_ready = df_active.dropna(subset=FEATURE_COLS).copy()
    logger.info(f'\nActive trials with complete features : {len(active_ready):,}')
    logger.info(f'Dropped due to missing features      : {len(df_active)-len(active_ready):,}')

    active_ready['success_probability'] = xgb_final.predict_proba(
        active_ready[FEATURE_COLS])[:, 1]
    active_ready['risk_score'] = 1 - active_ready['success_probability']
    active_ready['risk_label'] = pd.cut(
        active_ready['risk_score'],
        bins=[0, 0.30, 0.50, 0.75, 1.0],
        labels=['Low Risk','Moderate Risk','High Risk','Critical Risk']
    )

    # Risk profile summary
    rc = active_ready['risk_label'].value_counts()
    logger.info(f'\nRISK PROFILE — {DEPLOY_SPONSOR} ACTIVE PIPELINE')
    logger.info('=' * 55)
    for label in ['Critical Risk','High Risk','Moderate Risk','Low Risk']:
        if label in rc.index:
            count = rc[label]
            pct   = count / len(active_ready) * 100
            flag  = ' <- REVIEW THIS WEEK' if label == 'Critical Risk' else ''
            logger.info(f'  {label:<20} {count:>5,}  ({pct:.1f}%){flag}')
    logger.info(f'\n  Total active trials  : {len(active_ready):,}')
    logger.info(f'  Avg success prob     : {active_ready["success_probability"].mean()*100:.1f}%')

    # Top 10 Critical Risk
    critical = active_ready[
        active_ready['risk_label'] == 'Critical Risk'
    ].sort_values('risk_score', ascending=False)
    if len(critical) > 0:
        logger.info(f'\nTOP 10 CRITICAL RISK TRIALS — {DEPLOY_SPONSOR}')
        logger.info('=' * 55)
        for _, row in critical.head(10).iterrows():
            nct   = row.get('NCT', 'N/A')
            phase = row.get('Phase', 'N/A')
            cond  = str(row.get('Condition', 'N/A'))[:30]
            score = row['risk_score']
            logger.info(f'  {nct}  {phase}  {cond:<30}  {score:.3f}')

    # Save CSV
    active_out_cols = [
        'NCT','Sponsor','Phase','Status','Condition',
        'Start_Year','Enrollment','therapeutic_area',
        'success_probability','risk_score','risk_label'
    ]
    active_out_cols = [c for c in active_out_cols if c in active_ready.columns]
    sponsor_slug    = DEPLOY_SPONSOR.lower().replace(' ','_')
    csv_path        = os.path.join(OUTPUT_DIR, f'{sponsor_slug}_active_alerts.csv')
    active_ready[active_out_cols].to_csv(csv_path, index=False)
    logger.info(f'\n✓ CSV saved: {sponsor_slug}_active_alerts.csv')
    logger.info(f'  This is the Monday morning dashboard input for {DEPLOY_SPONSOR}.')
    logger.info('\n✓ Active trial scoring complete.')

    return active_ready
