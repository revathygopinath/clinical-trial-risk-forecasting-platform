# ─────────────────────────────────────────────────────────────
# eda.py
# Saves all charts to outputs/
# ─────────────────────────────────────────────────────────────

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import OUTPUT_DIR

sns.set_palette('husl')
sns.set_style('whitegrid')


def _save(filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


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


def run_eda(df_raw: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Generate all 4 EDA charts. Save to outputs/.
    Mirrors notebook Section 1 exactly.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info('=' * 55)
    logger.info('STEP 2 — EXPLORATORY DATA ANALYSIS')
    logger.info('=' * 55)

    finished = df_raw[df_raw['Status'].isin(['Completed','Terminated','Withdrawn'])]
    success_rate = (finished['Status'] == 'Completed').mean() * 100
    failure_rate = 100 - success_rate
    logger.info(f'Finished trials    : {len(finished):,}')
    logger.info(f'Success rate       : {success_rate:.1f}%')
    logger.info(f'Failure rate       : {failure_rate:.1f}%')
    logger.info('Class imbalance → using AUC-ROC not accuracy.')
    logger.info('')

    # ── Chart 1 — Trial Status Distribution ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc     = df_raw['Status'].value_counts()
    colors = ['#2ecc71' if s == 'Completed'
              else '#e74c3c' if s in ['Terminated','Withdrawn']
              else '#3498db' for s in sc.index]
    axes[0].bar(sc.index, sc.values, color=colors, edgecolor='white')
    axes[0].set_title('Trial Status Distribution', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].set_ylabel('Number of Trials')
    for i, v in enumerate(sc.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontsize=9, fontweight='bold')
    pc = ['#2ecc71','#e74c3c','#3498db','#f39c12','#9b59b6','#1abc9c','#e67e22','#95a5a6']
    axes[1].pie(sc.values, labels=sc.index, autopct='%1.1f%%',
                colors=pc[:len(sc)], startangle=140,
                wedgeprops={'edgecolor':'white','linewidth':1.5})
    axes[1].set_title('Status Percentage', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save('01_status.png')
    logger.info('✓ Chart 1 saved: 01_status.png')

    # ── Chart 2 — Phase Distribution and Completion Rates ────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pc2 = df_raw['Phase'].value_counts()
    axes[0].barh(pc2.index, pc2.values,
                 color=['#3498db','#2ecc71','#e74c3c','#f39c12',
                        '#9b59b6','#1abc9c','#e67e22'][:len(pc2)],
                 edgecolor='white')
    axes[0].set_title('Number of Trials by Phase', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of Trials')
    for i, v in enumerate(pc2.values):
        axes[0].text(v + 20, i, str(v), va='center', fontsize=9, fontweight='bold')

    finished_ph = df_raw[df_raw['Status'].isin(['Completed','Terminated','Withdrawn'])].copy()
    finished_ph['success'] = (finished_ph['Status'] == 'Completed').astype(int)
    mp = ['Phase 1','Phase 2','Phase 3','Phase 4']
    ps = finished_ph[finished_ph['Phase'].isin(mp)].groupby('Phase')['success'].mean() * 100
    ps = ps.reindex(mp)
    bc = ['#e74c3c' if v < 85 else '#2ecc71' for v in ps.values]
    axes[1].bar(ps.index, ps.values, color=bc, edgecolor='white')
    axes[1].set_title('Completion Rate by Phase', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 105)
    axes[1].axhline(y=ps.mean(), color='black', linestyle='--', alpha=0.5,
                    label='Average: ' + str(round(ps.mean(), 1)) + '%')
    axes[1].legend(fontsize=9)
    for i, v in enumerate(ps.values):
        axes[1].text(i, v + 1, str(round(v, 1)) + '%',
                     ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save('02_phases.png')
    logger.info('✓ Chart 2 saved: 02_phases.png')
    logger.info('Phase completion rates:')
    for phase, rate in ps.items():
        diff = rate - ps.mean()
        logger.info(f'  {phase}: {rate:.1f}%  '
                    f'({"above" if diff > 0 else "below"} avg by {abs(diff):.1f}%)')
    logger.info('')

    # ── Chart 3 — Therapeutic Area Risk ──────────────────────
    df_raw['Therapeutic_Area'] = df_raw['Condition'].apply(_map_ta)
    finished_ta = df_raw[df_raw['Status'].isin(['Completed','Terminated','Withdrawn'])].copy()
    finished_ta['success'] = (finished_ta['Status'] == 'Completed').astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    tac = df_raw['Therapeutic_Area'].value_counts()
    axes[0].bar(tac.index, tac.values,
                color=['#e74c3c','#3498db','#2ecc71','#f39c12',
                       '#9b59b6','#1abc9c','#e67e22','#95a5a6'],
                edgecolor='white')
    axes[0].set_title('Trial Volume by Therapeutic Area', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].set_ylabel('Number of Trials')
    for i, v in enumerate(tac.values):
        axes[0].text(i, v + 20, str(v), ha='center', fontsize=8, fontweight='bold')

    tas     = finished_ta.groupby('Therapeutic_Area')['success'].mean() * 100
    tas     = tas.sort_values()
    overall = finished_ta['success'].mean() * 100
    bc2     = ['#e74c3c' if v < overall else '#2ecc71' for v in tas.values]
    axes[1].barh(tas.index, tas.values, color=bc2, edgecolor='white')
    axes[1].axvline(x=overall, color='black', linestyle='--', alpha=0.6,
                    label='Overall avg: ' + str(round(overall, 1)) + '%')
    axes[1].set_title('Completion Rate by Therapeutic Area', fontsize=13, fontweight='bold')
    axes[1].set_xlim(0, 105)
    axes[1].legend(fontsize=9)
    for i, v in enumerate(tas.values):
        axes[1].text(v + 0.5, i, str(round(v, 1)) + '%', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    _save('03_therapeutic_areas.png')
    logger.info('✓ Chart 3 saved: 03_therapeutic_areas.png')
    logger.info('Risk ranking (lowest completion rate):')
    for rank, (ta, rate) in enumerate(tas.items(), 1):
        flag = 'HIGH RISK' if rate < overall else 'lower risk'
        logger.info(f'  {rank}. {ta:<22} {rate:.1f}%  [{flag}]')
    logger.info('')

    # ── Chart 4 — Time Trends and Enrollment Size ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    yearly = df_raw.groupby('Start_Year').size().reset_index(name='Count')
    yearly = yearly[(yearly['Start_Year'] >= 1995) & (yearly['Start_Year'] <= 2018)]
    axes[0].fill_between(yearly['Start_Year'], yearly['Count'], alpha=0.3, color='#3498db')
    axes[0].plot(yearly['Start_Year'], yearly['Count'],
                 color='#3498db', linewidth=2.5, marker='o', markersize=4)
    peak_year  = yearly.loc[yearly['Count'].idxmax(), 'Start_Year']
    peak_count = yearly['Count'].max()
    axes[0].axvline(x=peak_year, color='red', linestyle='--', alpha=0.7,
                    label='Peak: ' + str(peak_year) + ' (' + str(peak_count) + ' trials)')
    axes[0].set_title('Clinical Trials Started Per Year', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Trials Started')
    axes[0].legend(fontsize=9)

    finished_en = df_raw[df_raw['Status'].isin(['Completed','Terminated','Withdrawn'])].copy()
    finished_en['success'] = (finished_en['Status'] == 'Completed').astype(int)
    finished_en = finished_en[
        (finished_en['Enrollment'] > 0) & (finished_en['Enrollment'] <= 10000)
    ]
    finished_en['Bucket'] = pd.cut(
        finished_en['Enrollment'],
        bins=[0, 50, 200, 500, 1000, 10000],
        labels=['Tiny\n1-50','Small\n51-200','Med\n201-500','Large\n501-1k','VLarge\n>1k']
    )
    bs = finished_en.groupby('Bucket')['success'].mean() * 100
    axes[1].bar(bs.index, bs.values,
                color=['#e74c3c','#e67e22','#f1c40f','#2ecc71','#27ae60'],
                edgecolor='white')
    axes[1].set_title('Completion Rate by Trial Size', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 105)
    for i, v in enumerate(bs.values):
        axes[1].text(i, v + 1, str(round(v, 1)) + '%',
                     ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save('04_time_enrollment.png')
    logger.info('✓ Chart 4 saved: 04_time_enrollment.png')
    logger.info('Enrollment vs completion rate:')
    for bucket, rate in bs.items():
        logger.info(f'  {str(bucket):<18}: {rate:.1f}%')
    logger.info('')
    logger.info('✓ EDA complete. 4 charts saved.')
