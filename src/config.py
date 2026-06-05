# ─────────────────────────────────────────────────────────────
# config.py
# All constants in one place.
# Change anything here — entire pipeline updates automatically.
# ─────────────────────────────────────────────────────────────

import os

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'AERO-BirdsEye-Data.csv')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs')
LOG_DIR     = os.path.join(BASE_DIR, 'logs')

# ── Sponsor ───────────────────────────────────────────────────
# Change this to any sponsor name in the dataset.
# Entire pipeline filters to this sponsor for scoring + simulation.
# Model training always uses ALL sponsors for maximum signal.
DEPLOY_SPONSOR = 'Novartis'

# ── Data Filtering ────────────────────────────────────────────
DATA_START  = 2005   # remove noisy pre-2005 data (failure rate 0-7%)
DATA_END    = 2017   # remove post-2017 fast-completing bias
TRAIN_END   = 2013   # final train/test split point
TEST_END    = 2017

# ── Model ─────────────────────────────────────────────────────
RANDOM_STATE        = 42
N_ITER_SEARCH       = 50   # RandomizedSearchCV iterations
CV_FOLDS            = 3
# Force XGBoost initial model always — matches notebook exactly.
# Notebook always selected initial (tuning did not improve).
# Set True to guarantee notebook-identical numbers.
FORCE_INITIAL_MODEL = True

# ── Simulation ────────────────────────────────────────────────
N_SIMULATIONS   = 10_000
N_YEARS         = 5
FDA_APPROVAL    = 0.55   # historical Phase 3 completion → FDA approval rate

# TIME_FRACTION: probability of FDA approval within N_YEARS window
# from current phase. Already folds in time AND FDA approval rate.
# Phase 3: 0.40 (time) × 0.55 (FDA) = 0.22
# Phase 2: 0.08 — accounts for Phase 2 + Phase 3 + FDA in 5 yrs
# Phase 1: 0.02 — accounts for Phase 1 + 2 + 3 + FDA in 5 yrs
TIME_FRACTION   = {3: 0.22, 2: 0.08, 1: 0.02}

# Industry benchmark caps — prevent unrealistically high probabilities
PHASE_CAPS      = {1: 0.63, 2: 0.40, 3: 0.65}

# ── Business Thresholds ───────────────────────────────────────
THRESHOLD_QUARTERLY   = 0.30   # max recall quarterly screening
THRESHOLD_WEEKLY      = 0.75   # primary weekly operations (MAIN)
THRESHOLD_EMERGENCY   = 0.95   # board escalation max precision

# ── Feature Lists ─────────────────────────────────────────────
# Defined here as placeholder — actual list built in feature_engineering.py
# after one-hot encoding (TA columns vary by dataset)
CORE_FEATURES = [
    'phase_num',
    'enrollment_log',
    'enrollment_bucket',
    'sponsor_success_rate',
    'start_decade',
    'phase_x_enrollment',
    'phase_x_bigpharma',
]

# ── Reference Lists ───────────────────────────────────────────
BIG_PHARMA = [
    'GSK', 'Novartis', 'Pfizer', 'Merck', 'Sanofi',
    'JNJ', 'Roche', 'Bayer', 'AbbVie', 'Gilead'
]

ACTIVE_STATUSES = [
    'Recruiting',
    'Active, not recruiting',
    'Not yet recruiting',
    'Enrolling by invitation',
]

FINISHED_STATUSES = ['Completed', 'Terminated', 'Withdrawn']

PHASE_MAP = {
    'Phase 1': 1, 'Early Phase 1': 1, 'Phase 1/Phase 2': 1,
    'Phase 2': 2, 'Phase 2/Phase 3': 2,
    'Phase 3': 3, 'Phase 4': 4
}
