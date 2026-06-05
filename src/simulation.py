# ─────────────────────────────────────────────────────────────
# simulation.py
# Monte Carlo pipeline forecast for active sponsor trials.
# ─────────────────────────────────────────────────────────────

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List

from src.config import (
    OUTPUT_DIR, DEPLOY_SPONSOR,
    N_SIMULATIONS, N_YEARS,
    TIME_FRACTION, PHASE_CAPS, PHASE_MAP
)


def run_simulation(
    active_ready: pd.DataFrame,
    logger: logging.Logger
) -> None:
    """
    Monte Carlo simulation on active sponsor trials.
    Uses XGBoost survival probabilities × TIME_FRACTION.
    TIME_FRACTION already includes FDA 55% approval rate.
    No separate FDA multiplier in the loop.
    Saves 3 CSVs and 1 chart to outputs/
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info('=' * 60)
    logger.info('STEP 8 — MONTE CARLO PIPELINE FORECAST')
    logger.info(f'Sponsor        : {DEPLOY_SPONSOR}')
    logger.info(f'Simulations    : {N_SIMULATIONS:,}')
    logger.info(f'Forecast window: {N_YEARS} years')
    logger.info('=' * 60)

    if len(active_ready) == 0:
        logger.warning('No active trials to simulate. Skipping.')
        return

    # Prepare simulation data
    sim_data = active_ready.copy()
    sim_data['phase_num'] = sim_data['Phase'].map(PHASE_MAP).fillna(0).astype(int)
    sim_data = sim_data[sim_data['phase_num'].isin([1, 2, 3])].copy()

    # Cap survival probability at industry benchmarks
    sim_data['survival_prob'] = sim_data.apply(
        lambda r: min(r['success_probability'], PHASE_CAPS.get(r['phase_num'], 0.65)),
        axis=1
    )

    # effective_prob = survival_prob × TIME_FRACTION
    # TIME_FRACTION already folds in time AND FDA 55% approval rate.
    # Phase 3: 0.40 (time) × 0.55 (FDA) = 0.22
    # Phase 2: 0.08 — already means approval probability within 5 yrs
    # Phase 1: 0.02 — already means approval probability within 5 yrs
    # NO separate FDA multiplier needed in simulation loop.
    sim_data['effective_prob'] = sim_data.apply(
        lambda r: r['survival_prob'] * TIME_FRACTION.get(r['phase_num'], 0.22),
        axis=1
    )

    p1 = sim_data[sim_data['phase_num'] == 1]
    p2 = sim_data[sim_data['phase_num'] == 2]
    p3 = sim_data[sim_data['phase_num'] == 3]

    logger.info('\nActive trials entering simulation:')
    logger.info(f'  Phase 1 : {len(p1):>4} trials  '
                f'survival prob: {p1["survival_prob"].mean():.2f}  '
                f'approval prob within {N_YEARS}yrs: {p1["effective_prob"].mean():.3f}')
    logger.info(f'  Phase 2 : {len(p2):>4} trials  '
                f'survival prob: {p2["survival_prob"].mean():.2f}  '
                f'approval prob within {N_YEARS}yrs: {p2["effective_prob"].mean():.3f}')
    logger.info(f'  Phase 3 : {len(p3):>4} trials  '
                f'survival prob: {p3["survival_prob"].mean():.2f}  '
                f'approval prob within {N_YEARS}yrs: {p3["effective_prob"].mean():.3f}')
    logger.info(f'\nNOTE: effective_prob = survival_prob × TIME_FRACTION')
    logger.info(f'      TIME_FRACTION already includes FDA 55% approval rate.')
    logger.info(f'      No separate FDA multiplier applied in simulation loop.')

    p1_probs = p1['effective_prob'].values
    p2_probs = p2['effective_prob'].values
    p3_probs = p3['effective_prob'].values

    # ── Monte Carlo Loop ──────────────────────────────────────
    logger.info(f'\nRunning {N_SIMULATIONS:,} simulations...')
    np.random.seed(42)

    yearly_approvals = np.zeros((N_SIMULATIONS, N_YEARS))
    total_approvals  = np.zeros(N_SIMULATIONS)

    for sim in range(N_SIMULATIONS):

        # Phase 3 → approvals in Years 1 and 2
        # effective_prob = survival × TIME_FRACTION[3] (already includes FDA)
        p3_approvals = (np.random.random(len(p3_probs)) < p3_probs).sum()
        y1 = np.random.binomial(p3_approvals, 0.55)
        y2 = p3_approvals - y1
        yearly_approvals[sim, 0] += y1
        yearly_approvals[sim, 1] += y2

        # Phase 2 → approvals in Years 3 and 4
        # effective_prob already accounts for Phase 3 + FDA
        p2_approvals = (np.random.random(len(p2_probs)) < p2_probs).sum()
        y3 = np.random.binomial(p2_approvals, 0.55)
        y4 = p2_approvals - y3
        yearly_approvals[sim, 2] += y3
        yearly_approvals[sim, 3] += y4

        # Phase 1 → approvals in Year 5
        # effective_prob already accounts for all phases + FDA
        p1_approvals = (np.random.random(len(p1_probs)) < p1_probs).sum()
        yearly_approvals[sim, 4] += p1_approvals

        total_approvals[sim] = yearly_approvals[sim].sum()

    # ── Results ───────────────────────────────────────────────
    p10  = int(np.percentile(total_approvals, 10))
    p50  = int(np.percentile(total_approvals, 50))
    p90  = int(np.percentile(total_approvals, 90))
    mean = round(total_approvals.mean(), 1)

    logger.info('\n' + '=' * 60)
    logger.info(f'RESULTS — {DEPLOY_SPONSOR} {N_YEARS}-YEAR PIPELINE FORECAST')
    logger.info('=' * 60)
    logger.info(f'  Mean estimated approvals   : {mean}')
    logger.info(f'  Pessimistic  (P10)         : {p10}')
    logger.info(f'  Base Case    (P50)         : {p50}')
    logger.info(f'  Optimistic   (P90)         : {p90}')
    logger.info(f'  Min (worst simulation)     : {int(total_approvals.min())}')
    logger.info(f'  Max (best simulation)      : {int(total_approvals.max())}')
    logger.info('')
    logger.info('NOTE: Output = ESTIMATED APPROVALS, not commercial launches.')
    logger.info('      Model predicts phase completion probability.')
    logger.info('      Commercial launch depends on additional factors')
    logger.info('      outside this model scope.')
    logger.info('')
    logger.info('YEAR-BY-YEAR BASE CASE (P50):')
    for yr in range(N_YEARS):
        med  = int(np.median(yearly_approvals[:, yr]))
        p10y = int(np.percentile(yearly_approvals[:, yr], 10))
        p90y = int(np.percentile(yearly_approvals[:, yr], 90))
        source = ('Phase 3 completing' if yr < 2
                  else 'Phase 2 survivors' if yr < 4
                  else 'Phase 1 survivors')
        logger.info(f'  Year {yr+1}: {med:>2} approvals  '
                    f'[P10={p10y}, P90={p90y}]  ({source})')

    # ── Save CSVs ─────────────────────────────────────────────
    slug = DEPLOY_SPONSOR.lower().replace(' ', '_')

    pd.DataFrame({
        'simulation'     : range(N_SIMULATIONS),
        'total_approvals': total_approvals.astype(int)
    }).to_csv(os.path.join(OUTPUT_DIR, f'{slug}_forecast_approvals.csv'), index=False)

    pd.DataFrame([{
        'year': yr + 1,
        'p10' : int(np.percentile(yearly_approvals[:, yr], 10)),
        'p50' : int(np.median(yearly_approvals[:, yr])),
        'p90' : int(np.percentile(yearly_approvals[:, yr], 90)),
        'mean': round(yearly_approvals[:, yr].mean(), 2)
    } for yr in range(N_YEARS)]).to_csv(
        os.path.join(OUTPUT_DIR, f'{slug}_forecast_phases.csv'), index=False)

    pd.DataFrame({
        'scenario'       : ['Pessimistic (P10)','Base Case (P50)','Optimistic (P90)'],
        'total_approvals': [p10, p50, p90]
    }).to_csv(os.path.join(OUTPUT_DIR, f'{slug}_scenarios.csv'), index=False)

    logger.info(f'\nCSVs saved:')
    logger.info(f'  outputs/{slug}_forecast_approvals.csv')
    logger.info(f'  outputs/{slug}_forecast_phases.csv')
    logger.info(f'  outputs/{slug}_scenarios.csv')

    # ── Charts ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Pipeline Forecast — Monte Carlo ({N_SIMULATIONS:,} simulations)\n'
        f'effective_prob = phase survival × time fraction × 55% FDA approval rate\n'
        f'Output = Estimated Approvals (not commercial launches)',
        fontsize=11, fontweight='bold'
    )

    axes[0].hist(total_approvals,
                 bins=range(0, int(total_approvals.max()) + 2),
                 color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(p10, color='red',   linestyle='--', lw=2, label=f'P10 = {p10}')
    axes[0].axvline(p50, color='blue',  linestyle='--', lw=2, label=f'P50 = {p50}')
    axes[0].axvline(p90, color='green', linestyle='--', lw=2, label=f'P90 = {p90}')
    axes[0].set_title('Total Estimated Approvals\n(5-Year Distribution)')
    axes[0].set_xlabel('Estimated Approvals')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    years  = [f'Year {i+1}' for i in range(N_YEARS)]
    p10_yr = [int(np.percentile(yearly_approvals[:, i], 10)) for i in range(N_YEARS)]
    p50_yr = [int(np.median(yearly_approvals[:, i]))          for i in range(N_YEARS)]
    p90_yr = [int(np.percentile(yearly_approvals[:, i], 90)) for i in range(N_YEARS)]

    axes[1].plot(years, p50_yr, 'b-o', lw=2,   label='Base Case (P50)')
    axes[1].fill_between(years, p10_yr, p90_yr, alpha=0.15, color='blue', label='P10-P90 band')
    axes[1].plot(years, p10_yr, 'r--', lw=1.5, label='Pessimistic (P10)')
    axes[1].plot(years, p90_yr, 'g--', lw=1.5, label='Optimistic (P90)')
    axes[1].set_title('Year-by-Year Estimated Approvals')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Estimated Approvals')
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(bottom=0)

    bars = axes[2].bar(
        ['Pessimistic\n(P10)', 'Base Case\n(P50)', 'Optimistic\n(P90)'],
        [p10, p50, p90],
        color=['#d9534f','#5bc0de','#5cb85c'],
        edgecolor='white', width=0.5
    )
    for bar, val in zip(bars, [p10, p50, p90]):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(val), ha='center', va='bottom',
            fontweight='bold', fontsize=13
        )
    axes[2].set_title('Scenario Comparison\n(5-Year Total)')
    axes[2].set_ylabel('Estimated Approvals')
    axes[2].set_ylim(0, p90 + 3)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, f'{slug}_10_monte_carlo.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'✓ Chart saved: {slug}_10_monte_carlo.png')
    logger.info('\n✓ Simulation complete.')
