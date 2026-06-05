# ─────────────────────────────────────────────────────────────
# data_loader.py
# Load CSV, validate, print health check to terminal.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import logging
from src.config import DATA_PATH


def load_and_validate(logger: logging.Logger) -> pd.DataFrame:
    """
    Load the dataset and print a full health check to terminal.
    Returns df_raw unchanged — no filtering, no feature engineering.
    All downstream modules receive df_raw and do their own work.
    """
    logger.info('=' * 55)
    logger.info('STEP 1 — LOAD AND VALIDATE DATASET')
    logger.info('=' * 55)

    try:
        df_raw = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logger.error(f'Dataset not found at: {DATA_PATH}')
        logger.error('Place AERO-BirdsEye-Data.csv in the data/ folder.')
        raise

    logger.info(f'Total rows         : {df_raw.shape[0]:,}')
    logger.info(f'Total columns      : {df_raw.shape[1]}')
    logger.info(f'Date range         : {df_raw["Start_Year"].min()} to {df_raw["Start_Year"].max()}')
    logger.info(f'Unique sponsors    : {df_raw["Sponsor"].nunique():,}')
    logger.info(f'Unique conditions  : {df_raw["Condition"].nunique():,}')
    logger.info(f'Missing Phase      : {df_raw["Phase"].isnull().sum()} rows')
    logger.info('')

    logger.info('Status breakdown:')
    sv = df_raw['Status'].value_counts()
    for s, v in sv.items():
        logger.info(f'  {s:<35} {v:>6,}  ({v/len(df_raw)*100:.1f}%)')
    logger.info('')

    logger.info('Phase breakdown:')
    pv = df_raw['Phase'].value_counts()
    for p, v in pv.items():
        logger.info(f'  {p:<25} {v:>6,}  ({v/len(df_raw)*100:.1f}%)')
    logger.info('')

    logger.info('Top sponsors:')
    for sponsor, count in df_raw['Sponsor'].value_counts().head(10).items():
        logger.info(f'  {sponsor:<20} {count:>6,}')
    logger.info('')

    logger.info('✓ Dataset loaded and validated.')
    return df_raw
