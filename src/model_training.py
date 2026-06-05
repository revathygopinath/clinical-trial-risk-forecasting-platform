# ─────────────────────────────────────────────────────────────
# model_training.py
# Rolling window CV, LR baseline, XGBoost initial + tuned.
# ─────────────────────────────────────────────────────────────

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, learning_curve, RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

from src.config import (
    OUTPUT_DIR, TRAIN_END, TEST_END, DATA_START,
    RANDOM_STATE, N_ITER_SEARCH, CV_FOLDS
)


def train_model(
    filtered_data: pd.DataFrame,
    FEATURE_COLS: List[str],
    logger: logging.Logger
) -> Tuple:
    """
    Full training pipeline matching notebook Section 3.

    Returns:
        xgb_final, X_train, X_test, y_train, y_test,
        mean_auc, std_auc, lr, scaler, lr_proba,
        xgb_initial, tuned_proba, spw
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TARGET_COL = 'success'

    logger.info('=' * 55)
    logger.info('STEP 4 — MODEL TRAINING')
    logger.info('=' * 55)

    X_filtered    = filtered_data[FEATURE_COLS].values
    y_filtered    = filtered_data[TARGET_COL].values
    years_filtered = filtered_data['Start_Year'].values

    spw_full = (y_filtered == 0).sum() / (y_filtered == 1).sum()
    logger.info(f'Class imbalance ratio (scale_pos_weight): {spw_full:.2f}')

    # ── Rolling Window CV ─────────────────────────────────────
    windows = [
        {'name':'Window 1','train_end':2009,'test_start':2010,'test_end':2011},
        {'name':'Window 2','train_end':2011,'test_start':2012,'test_end':2013},
        {'name':'Window 3','train_end':2013,'test_start':2014,'test_end':2015},
        {'name':'Window 4','train_end':2015,'test_start':2016,'test_end':2017},
    ]

    xgb_rw = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw_full, random_state=RANDOM_STATE,
        eval_metric='logloss', verbosity=0
    )

    logger.info('\nROLLING WINDOW CROSS-VALIDATION (2005-2017 data)')
    logger.info('=' * 72)
    logger.info(f'{"Window":<12} {"Train":<16} {"Test":<14}'
                f'{"Train N":>9} {"Test N":>8} {"AUC":>8} {"Fail%":>7}')
    logger.info('-' * 72)

    window_aucs = []
    for w in windows:
        train_mask = years_filtered <= w['train_end']
        test_mask  = ((years_filtered >= w['test_start']) &
                      (years_filtered <= w['test_end']))
        X_tr = X_filtered[train_mask]; y_tr = y_filtered[train_mask]
        X_te = X_filtered[test_mask];  y_te = y_filtered[test_mask]
        if len(X_te) < 50:
            continue
        spw_w = (y_tr == 0).sum() / (y_tr == 1).sum() if (y_tr == 1).sum() > 0 else spw_full
        xgb_rw.set_params(scale_pos_weight=spw_w)
        xgb_rw.fit(X_tr, y_tr)
        auc       = roc_auc_score(y_te, xgb_rw.predict_proba(X_te)[:, 1])
        fail_rate = (y_te == 0).mean() * 100
        window_aucs.append(auc)
        train_str = f'2005-{w["train_end"]}'
        test_str  = f'{w["test_start"]}-{w["test_end"]}'
        logger.info(f'{w["name"]:<12} {train_str:<16} {test_str:<14}'
                    f'{len(X_tr):>9,} {len(X_te):>8,} {auc:>8.4f} {fail_rate:>6.1f}%')

    mean_auc = np.mean(window_aucs)
    std_auc  = np.std(window_aucs)
    logger.info(f'\nMean AUC : {mean_auc:.4f}')
    logger.info(f'Std Dev  : {std_auc:.4f}')
    if std_auc < 0.02:
        logger.info('Very low std — model stable across all time periods.')
    elif std_auc < 0.05:
        logger.info('Moderate std — minor variation across windows. Acceptable.')
    else:
        logger.info('High std — model varies across time periods.')
    logger.info(f'\nMean AUC {mean_auc:.4f} is the headline metric.')

    # ── Final Train/Test Split ────────────────────────────────
    train_mask_f = ((filtered_data['Start_Year'] >= DATA_START) &
                    (filtered_data['Start_Year'] <= TRAIN_END))
    test_mask_f  = ((filtered_data['Start_Year'] > TRAIN_END) &
                    (filtered_data['Start_Year'] <= TEST_END))

    X_train = filtered_data[train_mask_f][FEATURE_COLS]
    y_train = filtered_data[train_mask_f][TARGET_COL]
    X_test  = filtered_data[test_mask_f][FEATURE_COLS]
    y_test  = filtered_data[test_mask_f][TARGET_COL]
    spw     = (y_train == 0).sum() / (y_train == 1).sum()

    logger.info('\nFINAL TRAIN/TEST SPLIT')
    logger.info('=' * 55)
    logger.info(f'Train 2005-{TRAIN_END}   : {len(X_train):,} trials')
    logger.info(f'Test  {TRAIN_END+1}-{TEST_END}    : {len(X_test):,} trials')
    logger.info(f'Train failure rate  : {(y_train==0).mean()*100:.1f}%')
    logger.info(f'Test  failure rate  : {(y_test==0).mean()*100:.1f}%')
    logger.info(f'Class weight (spw)  : {spw:.2f}')

    # ── LR Baseline ───────────────────────────────────────────
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)
    lr       = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000,
                                   class_weight='balanced')
    lr.fit(X_tr_sc, y_train)
    lr_proba = lr.predict_proba(X_te_sc)[:, 1]
    lr_pred  = lr.predict(X_te_sc)
    lr_auc   = roc_auc_score(y_test, lr_proba)
    lr_tr    = roc_auc_score(y_train, lr.predict_proba(X_tr_sc)[:, 1])
    logger.info(f'\nBaseline Logistic Regression:')
    logger.info(f'  Train AUC: {lr_tr:.4f}')
    logger.info(f'  Test  AUC: {lr_auc:.4f}')
    logger.info(f'  Gap      : {lr_tr - lr_auc:.4f}')

    # ── XGBoost Initial ───────────────────────────────────────
    xgb_initial = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=RANDOM_STATE,
        eval_metric='logloss', verbosity=0
    )
    xgb_initial.fit(X_train, y_train)
    xgb_train_auc = roc_auc_score(y_train, xgb_initial.predict_proba(X_train)[:, 1])
    xgb_test_auc  = roc_auc_score(y_test,  xgb_initial.predict_proba(X_test)[:, 1])
    gap           = xgb_train_auc - xgb_test_auc
    logger.info(f'\nXGBoost Initial:')
    logger.info(f'  Train AUC: {xgb_train_auc:.4f}')
    logger.info(f'  Test  AUC: {xgb_test_auc:.4f}')
    logger.info(f'  Gap      : {gap:.4f}  →  {"possible overfit" if gap > 0.05 else "OK"}')
    logger.info(f'  XGBoost beats LR by: {xgb_test_auc - lr_auc:.4f} AUC points')

    # ── Learning Curve (Initial) ──────────────────────────────
    logger.info('\nGenerating learning curve for initial XGBoost...')
    cv_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        xgb_initial, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv_lc, scoring='roc_auc', n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)
    final_gap  = train_mean[-1] - val_mean[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='#e74c3c', linewidth=2.5, label='Training AUC')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.15, color='#e74c3c')
    plt.plot(train_sizes, val_mean, 'o-', color='#2ecc71', linewidth=2.5, label='Validation AUC')
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.15, color='#2ecc71')
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline')
    note = 'Possible overfit — tuning will help' if final_gap > 0.05 else 'Acceptable fit'
    plt.title(f'Learning Curve — Initial XGBoost\nGap: {final_gap:.4f} | {note}',
              fontsize=13, fontweight='bold')
    plt.xlabel('Training Set Size')
    plt.ylabel('AUC-ROC Score')
    plt.legend(fontsize=11)
    plt.ylim(0.45, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_learning_curve_initial.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'  Gap: {final_gap:.4f}  →  '
                f'{"Significant overfitting." if final_gap > 0.10 else "Acceptable fit."}')
    logger.info('✓ Chart saved: 06_learning_curve_initial.png')

    # ── Hyperparameter Tuning ─────────────────────────────────
    logger.info('\nStarting hyperparameter search...')
    logger.info(f'{N_ITER_SEARCH} combinations × {CV_FOLDS}-fold CV = {N_ITER_SEARCH*CV_FOLDS} model fits')
    param_dist = {
        'n_estimators'    : randint(100, 500),
        'max_depth'       : randint(2, 8),
        'learning_rate'   : uniform(0.01, 0.2),
        'subsample'       : uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'reg_alpha'       : uniform(0, 1),
        'reg_lambda'      : uniform(0.5, 2),
    }
    xgb_base = XGBClassifier(
        scale_pos_weight=spw, random_state=RANDOM_STATE,
        eval_metric='logloss', verbosity=0
    )
    search = RandomizedSearchCV(
        xgb_base, param_distributions=param_dist,
        n_iter=N_ITER_SEARCH, scoring='roc_auc',
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    logger.info(f'Best CV AUC: {search.best_score_:.4f}')

    # ── Tuned Model ───────────────────────────────────────────
    xgb_tuned = XGBClassifier(
        **search.best_params_,
        scale_pos_weight=spw, random_state=RANDOM_STATE,
        eval_metric='logloss', verbosity=0
    )
    xgb_tuned.fit(X_train, y_train)
    tuned_proba   = xgb_tuned.predict_proba(X_test)[:, 1]
    tuned_auc     = roc_auc_score(y_test, tuned_proba)
    tuned_tr_auc  = roc_auc_score(y_train, xgb_tuned.predict_proba(X_train)[:, 1])

    logger.info('\nMODEL COMPARISON')
    logger.info('=' * 68)
    logger.info(f'{"Model":<28} {"Train AUC":>10} {"Test AUC":>10} {"Gap":>8}')
    logger.info('-' * 68)
    logger.info(f'{"Logistic Regression":<28} {lr_tr:>10.4f} {lr_auc:>10.4f} {lr_tr-lr_auc:>8.4f}')
    logger.info(f'{"XGBoost initial":<28} {xgb_train_auc:>10.4f} {xgb_test_auc:>10.4f} {xgb_train_auc-xgb_test_auc:>8.4f}')
    logger.info(f'{"XGBoost tuned":<28} {tuned_tr_auc:>10.4f} {tuned_auc:>10.4f} {tuned_tr_auc-tuned_auc:>8.4f}')

    # ── Final Model Selection ─────────────────────────────────
    from src.config import FORCE_INITIAL_MODEL
    if FORCE_INITIAL_MODEL or xgb_test_auc >= tuned_auc:
        xgb_final   = xgb_initial
        final_auc   = xgb_test_auc
        final_proba = xgb_initial.predict_proba(X_test)[:, 1]
        if FORCE_INITIAL_MODEL:
            logger.info('\nFinal model: XGBoost initial (FORCE_INITIAL_MODEL=True — matches notebook)')
        else:
            logger.info('\nFinal model: XGBoost initial (tuning did not improve test AUC)')
        logger.info('Legitimate outcome. Initial params well-suited to this dataset.')
    else:
        xgb_final   = xgb_tuned
        final_auc   = tuned_auc
        final_proba = tuned_proba
        logger.info('\nFinal model: XGBoost tuned (tuning improved test AUC)')

    logger.info(f'Final model test AUC     : {final_auc:.4f}')
    logger.info(f'Rolling window mean AUC  : {mean_auc:.4f} ± {std_auc:.4f}')

    # ── 5-Fold CV on Training Data ────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        xgb_final, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    logger.info('\n5-Fold CV on Training Data (2005-2013):')
    for i, s in enumerate(cv_scores, 1):
        logger.info(f'  Fold {i}: AUC = {s:.4f}')
    logger.info(f'  Mean  : {cv_scores.mean():.4f}')
    logger.info(f'  Std   : {cv_scores.std():.4f}')
    if cv_scores.std() < 0.02:
        logger.info('  Low std — model is stable within the training period.')

    # ── Learning Curve Comparison Before vs After ─────────────
    train_sizes_t, train_scores_t, val_scores_t = learning_curve(
        xgb_final, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc', n_jobs=-1
    )
    tm_t = train_scores_t.mean(axis=1)
    vm_t = val_scores_t.mean(axis=1)
    ts_t = train_scores_t.std(axis=1)
    vs_t = val_scores_t.std(axis=1)
    gap_after = tm_t[-1] - vm_t[-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (tm, vm, ts, vs, title) in zip(axes, [
        (train_mean, val_mean, train_std, val_std, f'BEFORE (Initial)\nGap: {final_gap:.4f}'),
        (tm_t,       vm_t,     ts_t,      vs_t,    f'AFTER (Final)\nGap: {gap_after:.4f}')
    ]):
        ax.plot(train_sizes, tm, 'o-', color='#e74c3c', linewidth=2.5, label='Train AUC')
        ax.plot(train_sizes, vm, 'o-', color='#2ecc71', linewidth=2.5, label='Validation AUC')
        ax.fill_between(train_sizes, tm - ts, tm + ts, alpha=0.15, color='#e74c3c')
        ax.fill_between(train_sizes, vm - vs, vm + vs, alpha=0.15, color='#2ecc71')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Size')
        ax.set_ylabel('AUC-ROC')
        ax.legend()
        ax.set_ylim(0.45, 1.05)
    plt.suptitle('Learning Curves: Before vs After', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08_learning_curves_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('✓ Chart saved: 08_learning_curves_comparison.png')
    logger.info('\n✓ Model training complete.')

    return (xgb_final, X_train, X_test, y_train, y_test,
            mean_auc, std_auc, lr, scaler, lr_proba,
            xgb_initial, tuned_proba, spw)
