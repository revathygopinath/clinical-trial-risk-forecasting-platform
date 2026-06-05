# ─────────────────────────────────────────────────────────────
# logger.py
# Terminal output + timestamped log file.
# Every pipeline step prints to terminal AND writes to logs/
# ─────────────────────────────────────────────────────────────

import logging
import os
from datetime import datetime
from src.config import LOG_DIR


def get_logger(name: str = 'ClinicalEdge') -> logging.Logger:
    """
    Returns a logger that writes to both terminal and a timestamped log file.
    Call once in main.py. Pass the same logger to every module.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file    = os.path.join(LOG_DIR, f'pipeline_{timestamp}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Terminal handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f'Log file: {log_file}')
    return logger
