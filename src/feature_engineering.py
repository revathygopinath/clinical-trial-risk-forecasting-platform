# ─────────────────────────────────────────────────────────────
# feature_engineering.py
# ─────────────────────────────────────────────────────────────

import logging
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List

from src.config import (
    OUTPUT_DIR, BIG_PHARMA, FINISHED_STATUSES,
    DATA_START, DATA_END, CORE_FEATURES
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


def _cumulative_success_rate(x):
    """Time-aware sponsor success rate. No data leakage."""
    return x.shift(1).expanding().mean()


def build_features(
    df_raw: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], float, List[str]]:
    """
    Build all 14 features from df_raw.

    Returns:
        filtered_data  — 2005-2017 data with features, ready for train/test split
        ml_data        — full finished trials with features (for scoring all trials)
        FEATURE_COLS   — list of feature column names
        overall_rate   — overall dataset success rate (used as fallback)
        ta_feature_cols — one-hot TA column names (needed by scoring module)
    """
    logger.info('=' * 55)
    logger.info('STEP 3 — FEATURE ENGINEERING')
    logger.info('=' * 55)

    # Filter to finished trials only
    df = df_raw[df_raw['Status'].isin(FINISHED_STATUSES)].copy()
    df['success'] = (df['Status'] == 'Completed').astype(int)
    logger.info(f'Working dataset    : {len(df):,} finished trials')
    logger.info(f'Success rate       : {df["success"].mean()*100:.1f}%')
    logger.info(f'Failure rate       : {(1-df["success"].mean())*100:.1f}%')

    # Sort chronologically — required for time-aware sponsor feature
    df = df.sort_values('Start_Year').reset_index(drop=True)

    # Feature 1: Phase number (ordinal)
    phase_map = {
        'Phase 1': 1, 'Early Phase 1': 1, 'Phase 1/Phase 2': 1,
        'Phase 2': 2, 'Phase 2/Phase 3': 2,
        'Phase 3': 3, 'Phase 4': 4
    }
    df['phase_num'] = df['Phase'].map(phase_map).fillna(0).astype(int)

    # Features 2 and 3: Enrollment (log transform + bucket)
    p99 = df['Enrollment'].quantile(0.99)
    df['enrollment_log'] = np.log1p(df['Enrollment'].clip(upper=p99))
    df['enrollment_bucket'] = pd.cut(
        df['Enrollment'],
        bins=[-1, 0, 50, 200, 500, 1000, float('inf')],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)

    # Feature 4: Therapeutic Area (one-hot, drop Cardiovascular reference)
    df['therapeutic_area'] = df['Condition'].apply(_map_ta)
    ta_dummies      = pd.get_dummies(df['therapeutic_area'], prefix='ta', drop_first=True)
    df              = pd.concat([df, ta_dummies], axis=1)
    ta_feature_cols = ta_dummies.columns.tolist()

    logger.info('Therapeutic area one-hot columns:')
    for col in ta_feature_cols:
        logger.info(f'  {col}')
    logger.info('  (Cardiovascular is reference category — dropped)')

    # Feature 5: Big Pharma flag (used only in interaction)
    df['is_big_pharma_raw'] = df['Sponsor'].isin(BIG_PHARMA).astype(int)

    # Feature 6: Sponsor success rate — TIME-AWARE, NO LEAKAGE
    # shift(1) means each trial cannot see its own outcome
    # expanding().mean() uses only PAST outcomes of that sponsor
    overall_rate = df['success'].mean()
    df['sponsor_success_rate'] = (
        df.groupby('Sponsor')['success']
        .transform(_cumulative_success_rate)
        .fillna(overall_rate)
    )

    # Feature 7: Start decade (ordinal)
    df['start_decade'] = df['Start_Year'].apply(
        lambda y: 0 if y < 1990 else (1 if y < 2000 else (2 if y < 2010 else 3))
    )

    # Features 8 and 9: Interaction features
    df['phase_x_enrollment'] = df['phase_num'] * df['enrollment_log']
    df['phase_x_bigpharma']  = df['phase_num'] * df['is_big_pharma_raw']

    # Final feature list
    FEATURE_COLS = CORE_FEATURES + ta_feature_cols
    TARGET_COL   = 'success'

    ml_data = df[FEATURE_COLS + [TARGET_COL, 'Start_Year',
                                  'Sponsor', 'Phase', 'Condition',
                                  'Enrollment', 'therapeutic_area',
                                  'NCT']].dropna()

    logger.info(f'\nML-ready rows      : {len(ml_data):,}')
    logger.info(f'Features used      : {len(FEATURE_COLS)}')
    logger.info(f'Feature list       : {FEATURE_COLS}')

    # Leakage verification
    logger.info('\nLEAKAGE VERIFICATION — sponsor_success_rate')
    logger.info('=' * 55)
    gsk = df[df['Sponsor'] == 'GSK'][
        ['Sponsor','Start_Year','success','sponsor_success_rate']
    ].head(4)
    logger.info(f'Row 1 fallback rate = {round(overall_rate, 4)}')
    logger.info('Each row uses only PAST outcomes. No future leakage.')

    # Data filtering to 2005-2017
    filtered_data = ml_data[
        (ml_data['Start_Year'] >= DATA_START) &
        (ml_data['Start_Year'] <= DATA_END)
    ].copy()

    discarded_old    = (ml_data['Start_Year'] < DATA_START).sum()
    discarded_recent = (ml_data['Start_Year'] > DATA_END).sum()

    logger.info('\nDATA FILTERING')
    logger.info('=' * 55)
    logger.info(f'Full dataset       : {len(ml_data):,} trials')
    logger.info(f'Kept {DATA_START}-{DATA_END}          : {len(filtered_data):,} trials')
    logger.info(f'Removed pre-{DATA_START}    : {discarded_old:,}  (noisy, failure rate 0-7%)')
    logger.info(f'Removed post-{DATA_END}   : {discarded_recent:,}  (fast-completing bias, failure 49%)')
    logger.info(f'Failure rate full  : {(ml_data["success"]==0).mean()*100:.1f}%')
    logger.info(f'Failure rate 05-17 : {(filtered_data["success"]==0).mean()*100:.1f}%')

    # Correlation chart
    corr = ml_data.corr(numeric_only=True)[TARGET_COL].drop(
        [TARGET_COL,'Start_Year'], errors='ignore'
    ).sort_values()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(11, 7))
    bar_colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in corr.values]
    plt.barh(corr.index, corr.values, color=bar_colors, edgecolor='white')
    plt.axvline(x=0, color='black', linewidth=1)
    plt.title('Feature Correlation with Trial Success\n'
              'Green = positively predicts success | Red = predicts failure',
              fontsize=13, fontweight='bold')
    plt.xlabel('Pearson Correlation Coefficient')
    for i, v in enumerate(corr.values):
        plt.text(v + (0.002 if v >= 0 else -0.002), i, str(round(v, 4)),
                 va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_correlations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('\n✓ Correlation chart saved: 05_correlations.png')

    logger.info('\nTop 5 positively correlated:')
    for feat, val in corr[corr > 0].tail(5).items():
        logger.info(f'  {feat:<30}: +{val:.4f}')
    logger.info('Top 5 negatively correlated (predict failure):')
    for feat, val in corr[corr < 0].head(5).items():
        logger.info(f'  {feat:<30}:  {val:.4f}')
    logger.info('')
    logger.info('✓ Feature engineering complete.')

    return filtered_data, ml_data, FEATURE_COLS, overall_rate, ta_feature_cols
