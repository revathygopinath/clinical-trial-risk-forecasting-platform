"""
dashboard/app.py — v8
Clinical Trial Portfolio Intelligence & Risk Forecasting Platform
All v7 changes plus v8 updates applied.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

THEME = {
    "bg":"#FFFFFF","card_bg":"#F5F7FA","sidebar":"#1B3A6B",
    "text":"#1A1A2E","text_sub":"#4A5568","text_light":"#FFFFFF",
    "border":"#D1D5DB","divider":"#E5E7EB","accent":"#2E6DA4",
    "positive":"#1A6B3A","warning":"#B7800A","danger":"#C0392B","header":"#1B3A6B",
}

st.set_page_config(
    page_title='Clinical Trial Portfolio Intelligence & Risk Forecasting Platform',
    layout='wide', initial_sidebar_state='expanded')

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif!important;color:{THEME['text']};}}
.main,[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>.main{{background-color:{THEME['bg']}!important;}}
[data-testid="block-container"]{{background-color:{THEME['bg']}!important;padding-top:1.2rem!important;}}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div:first-child{{background-color:{THEME['sidebar']}!important;}}
[data-testid="stSidebar"] *{{color:#c8d6e5!important;}}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{{color:#ffffff!important;}}
[data-testid="stSidebar"] .stSelectbox label{{color:#8fa8c8!important;font-size:0.72rem!important;}}
[data-testid="stSidebar"] .stSelectbox>div>div{{background:#243d62!important;border:1px solid #2e4f7a!important;color:#c8d6e5!important;}}
.sb-section{{font-size:0.67rem;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#5a7fa0!important;padding:14px 20px 5px 20px;display:block;}}
.sb-title{{font-size:1.05rem;font-weight:700;color:#ffffff!important;padding:22px 20px 1px 20px;display:block;line-height:1.2;}}
.sb-sub{{font-size:0.72rem;color:#5a7fa0!important;padding:0 20px 14px 20px;display:block;}}
.sb-divider{{border:none;border-top:1px solid #243d62;margin:6px 18px;}}
.sb-meta{{font-size:0.72rem;color:#5a7fa0!important;padding:3px 20px;line-height:1.7;}}
.kpi{{background:{THEME['card_bg']};border-radius:8px;padding:16px 18px 14px 18px;border-left:4px solid {THEME['accent']};box-shadow:0 1px 3px rgba(0,0,0,0.07);margin-bottom:2px;}}
.kpi.red{{border-left-color:{THEME['danger']};}}
.kpi.green{{border-left-color:{THEME['positive']};}}
.kpi.orange{{border-left-color:{THEME['warning']};}}
.kpi.slate{{border-left-color:#64748b;}}
.kpi-val{{font-size:1.9rem;font-weight:700;color:{THEME['text']};line-height:1.1;}}
.kpi-lbl{{font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.09em;color:{THEME['text_sub']};margin-top:5px;}}
.kpi-sub{{font-size:0.75rem;font-weight:500;margin-top:2px;}}
.kpi-sub.red{{color:{THEME['danger']};}}
.kpi-sub.green{{color:{THEME['positive']};}}
.kpi-sub.orange{{color:{THEME['warning']};}}
.kpi-sub.slate{{color:#64748b;}}
.sec-hdr{{font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.11em;color:{THEME['text_sub']};padding:12px 0 7px 0;border-bottom:1px solid {THEME['divider']};margin-bottom:10px;}}
.pg-title{{font-size:1.6rem;font-weight:700;color:{THEME['header']};letter-spacing:-0.01em;margin-bottom:1px;line-height:1.2;}}
.pg-sub{{font-size:0.86rem;color:{THEME['text_sub']};margin-bottom:16px;}}
.card{{background:{THEME['card_bg']};border-radius:8px;padding:20px 22px;border:1px solid {THEME['border']};box-shadow:0 1px 2px rgba(0,0,0,0.05);}}
.card-title{{font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:{THEME['accent']};margin-bottom:8px;}}
.card-text{{font-size:0.83rem;color:{THEME['text']};line-height:1.65;}}
.note-blue{{background:#EFF6FF;border-left:4px solid {THEME['accent']};border-radius:6px;padding:11px 15px;font-size:0.8rem;color:#1e3a5f;margin-top:10px;line-height:1.6;}}
.note-amber{{background:#FFFBEB;border-left:4px solid {THEME['warning']};border-radius:6px;padding:11px 15px;font-size:0.8rem;color:#78350f;margin-top:10px;line-height:1.6;}}
.action-panel{{background:#1B3A6B;border-radius:8px;padding:18px 22px;color:#ffffff;margin-top:8px;}}
.action-title{{font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#93c5fd;margin-bottom:10px;}}
.action-item{{font-size:0.84rem;color:#e2e8f0;margin:5px 0;line-height:1.5;}}
.step-row{{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0;}}
.step{{background:{THEME['accent']};color:white;border-radius:6px;padding:6px 14px;font-size:0.78rem;font-weight:600;text-align:center;}}
.wf-bar{{background:{THEME['card_bg']};border-radius:6px;padding:12px 16px;margin:4px 0;border:1px solid {THEME['border']};}}
.wf-label{{font-size:0.82rem;color:{THEME['text_sub']};font-weight:500;}}
.wf-val{{font-size:1.1rem;font-weight:700;color:{THEME['text']};}}
#MainMenu,footer,header{{visibility:hidden;}}
.block-container{{padding-bottom:2rem;}}
</style>
""", unsafe_allow_html=True)


# ── Chart helper ──────────────────────────────────────────────
DARK = '#1A1A2E'   # forced dark text for all chart labels

def pgo(fig, height=300, legend=False, margin=None):
    m = margin or dict(t=20, b=20, l=8, r=8)
    fig.update_layout(
        height=height, margin=m,
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(family='Inter', color=DARK, size=11),
        showlegend=legend,
        xaxis=dict(gridcolor='#E5E7EB', linecolor='#D1D5DB',
                   tickfont=dict(color=DARK, size=11),
                   title_font=dict(color=DARK, size=12)),
        yaxis=dict(gridcolor='#E5E7EB', linecolor='#D1D5DB',
                   tickfont=dict(color=DARK, size=11),
                   title_font=dict(color=DARK, size=12)),
    )
    return fig


# ── Pipeline ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_pipeline():
    from src.config import OUTPUT_DIR, LOG_DIR, DEPLOY_SPONSOR
    from src.data_loader import load_and_validate
    from src.feature_engineering import build_features
    from src.model_training import train_model
    from src.evaluation import run_evaluation
    from src.scoring import score_all_trials, score_active_trials
    from src.simulation import run_simulation
    from sklearn.metrics import roc_auc_score
    import logging

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger('CE_Dash')
    logger.setLevel(logging.WARNING)

    df_raw = load_and_validate(logger)
    filtered_data, ml_data, FEATURE_COLS, overall_rate, ta_cols = build_features(df_raw, logger)
    (xgb_final, X_train, X_test, y_train, y_test,
     mean_auc, std_auc, lr, scaler, lr_proba,
     xgb_initial, tuned_proba, spw) = train_model(filtered_data, FEATURE_COLS, logger)
    xgb_final_auc = roc_auc_score(y_test, xgb_final.predict_proba(X_test)[:, 1])
    failure_proba, shap_values, _ = run_evaluation(
        xgb_final, xgb_initial, lr, X_train, X_test, y_train, y_test,
        lr_proba, tuned_proba, xgb_final_auc, FEATURE_COLS, logger)
    df_scored    = score_all_trials(ml_data, xgb_final, FEATURE_COLS, logger)
    active_ready = score_active_trials(
        df_raw, xgb_final, FEATURE_COLS, ta_cols, overall_rate, ml_data, logger)
    run_simulation(active_ready, logger)

    slug = DEPLOY_SPONSOR.lower().replace(' ', '_')
    def _csv(n):
        p = os.path.join(OUTPUT_DIR, n)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    scenarios = _csv(f'{slug}_scenarios.csv')
    phases_df = _csv(f'{slug}_forecast_phases.csv')
    approvals = _csv(f'{slug}_forecast_approvals.csv')

    shap_df = _csv('shap_importance.csv')
    if shap_df.empty and shap_values is not None:
        shap_df = pd.DataFrame({
            'feature': FEATURE_COLS,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)

    # Historical Novartis per-phase rates
    nov_finished = df_raw[
        (df_raw['Sponsor'] == DEPLOY_SPONSOR) &
        (df_raw['Status'].isin(['Completed','Terminated','Withdrawn']))
    ].copy()
    nov_finished['success'] = (nov_finished['Status'] == 'Completed').astype(int)
    hist_phase = {}
    for ph in ['Phase 1','Phase 2','Phase 3']:
        g = nov_finished[nov_finished['Phase'] == ph]
        hist_phase[ph] = round(g['success'].mean()*100, 1) if len(g) else 0.0

    # Predicted per-phase from active
    pred_phase = {}
    for ph in ['Phase 1','Phase 2','Phase 3']:
        g = active_ready[active_ready['Phase'] == ph] if len(active_ready) else pd.DataFrame()
        pred_phase[ph] = round(g['success_probability'].mean()*100, 1) if len(g) else 0.0

    # Phase-level trends for all phases
    phase_trends = {}
    for ph in ['Phase 1','Phase 2','Phase 3']:
        g = nov_finished[nov_finished['Phase'] == ph].copy()
        g['period'] = (g['Start_Year'] // 5) * 5
        t = g.groupby('period')['success'].agg(['mean','count']).reset_index()
        t.columns = ['period','rate','count']
        t['rate']  = (t['rate']*100).round(1)
        t['label'] = t['period'].astype(str)+'-'+(t['period']+4).astype(str).str[-2:]
        phase_trends[ph] = t[t['count'] >= 5].reset_index(drop=True)

    # Active p2 avg start year
    p2_active = active_ready[active_ready['Phase']=='Phase 2'] if len(active_ready) else pd.DataFrame()
    active_p2_year = round(p2_active['Start_Year'].mean(), 0) if len(p2_active) else 2016.0

    # Forecast P10/P50/P90
    p10, p50, p90 = 0, 0, 0
    if not scenarios.empty:
        p10 = int(scenarios[scenarios['scenario'].str.contains('P10')]['total_approvals'].values[0])
        p50 = int(scenarios[scenarios['scenario'].str.contains('P50')]['total_approvals'].values[0])
        p90 = int(scenarios[scenarios['scenario'].str.contains('P90')]['total_approvals'].values[0])

    # Highest risk phase
    worst_phase, worst_pred = 'Phase 2', 100.0
    for ph in ['Phase 1','Phase 2','Phase 3']:
        if pred_phase.get(ph, 100) < worst_pred:
            worst_pred  = pred_phase[ph]
            worst_phase = ph

    # Completion stats
    finished = df_raw[df_raw['Status'].isin(['Completed','Terminated','Withdrawn'])]
    hist_completion = round((finished['Status']=='Completed').mean()*100, 1)
    pred_completion = round(active_ready['success_probability'].mean()*100, 1) if len(active_ready) else 0.0
    completion_gap  = round(pred_completion - hist_completion, 1)

    return dict(
        active_ready=active_ready, df_scored=df_scored,
        mean_auc=mean_auc, std_auc=std_auc, xgb_final_auc=xgb_final_auc,
        scenarios=scenarios, phases_df=phases_df, approvals=approvals,
        shap_df=shap_df, shap_values=shap_values, FEATURE_COLS=FEATURE_COLS,
        hist_phase=hist_phase, pred_phase=pred_phase,
        phase_trends=phase_trends, active_p2_year=active_p2_year,
        hist_completion=hist_completion,
        pred_completion=pred_completion, completion_gap=completion_gap,
        worst_phase=worst_phase, worst_pred=worst_pred,
        p10=p10, p50=p50, p90=p90,
        DEPLOY_SPONSOR=DEPLOY_SPONSOR,
        total_trials=len(df_raw),
        df_raw_ref=df_raw,
    )


# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar(active_ready):
    st.sidebar.markdown('<span class="sb-title">ClinicalEdge</span>', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="sb-sub">Portfolio Intelligence & Risk Forecasting</span>', unsafe_allow_html=True)
    st.sidebar.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="sb-section">Navigation</span>', unsafe_allow_html=True)
    pages = ['Home','Executive Portfolio Overview',
             'Active Trial Monitor','Portfolio Forecast & Scenario Planning']
    page = st.sidebar.radio('nav', pages, index=0,
                             label_visibility='collapsed', key='nav_radio')
    st.sidebar.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="sb-section">Filters</span>', unsafe_allow_html=True)
    phases = ['All'] + sorted(active_ready['Phase'].dropna().unique().tolist())
    sel_phase = st.sidebar.selectbox('Phase', phases, key='f_phase')
    tas = ['All'] + sorted(active_ready['therapeutic_area'].dropna().unique().tolist())
    sel_ta = st.sidebar.selectbox('Therapeutic Area', tas, key='f_ta')
    risk_levels = ['All','Critical Risk','High Risk','Moderate Risk','Low Risk']
    sel_risk = st.sidebar.selectbox('Risk Level', risk_levels, key='f_risk')
    st.sidebar.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="sb-section">Output Status</span>', unsafe_allow_html=True)
    for item in ['Risk Scores','SHAP Importance','Forecast CSVs','Active Alerts']:
        st.sidebar.markdown(f'<span class="sb-meta">&#10003; {item}</span>', unsafe_allow_html=True)
    st.sidebar.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="sb-section">Dataset</span>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<span class="sb-meta">AERO-BirdsEye Dataset<br>'
        '13,748 Trials | 10 Sponsors<br>'
        'Training Window: 2005-2017</span>', unsafe_allow_html=True)
    df = active_ready.copy()
    if sel_phase != 'All': df = df[df['Phase'] == sel_phase]
    if sel_ta    != 'All': df = df[df['therapeutic_area'] == sel_ta]
    if sel_risk  != 'All': df = df[df['risk_label'] == sel_risk]
    return page, df


def kpi(val, label, sub='', cls='blue'):
    return (f'<div class="kpi {cls}"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-lbl">{label}</div>'
            + (f'<div class="kpi-sub {cls}">{sub}</div>' if sub else '')
            + '</div>')


LABEL_MAP = {
    'enrollment_log':'Enrollment Size','enrollment_bucket':'Enrollment Category',
    'phase_x_enrollment':'Phase x Enrollment','phase_x_bigpharma':'Phase x Sponsor Tier',
    'sponsor_success_rate':'Sponsor Track Record','phase_num':'Trial Phase',
    'start_decade':'Era / Start Decade','ta_Oncology':'Oncology Indication',
    'ta_Neurology':'Neurology Indication','ta_Immunology':'Immunology Indication',
    'ta_Infectious Disease':'Infectious Disease','ta_Metabolic':'Metabolic',
    'ta_Respiratory':'Respiratory','ta_Other':'Other Indication',
}


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════
def page_home(data):
    ar   = data['active_ready']
    p10, p50, p90 = data['p10'], data['p50'], data['p90']
    crit = int((ar['risk_label']=='Critical Risk').sum()) if len(ar) else 0
    hp   = data['hist_phase']

    st.markdown(
        '<div class="pg-title">Clinical Trial Portfolio Intelligence</div>'
        '<div style="font-size:1.1rem;font-weight:600;color:#2E6DA4;margin-bottom:4px">'
        'Risk Forecasting Platform</div>'
        '<div class="pg-sub">End-to-end ML system for clinical trial risk scoring and pipeline forecasting</div>',
        unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:1px solid #E5E7EB;margin:6px 0 16px 0">', unsafe_allow_html=True)

    # ── 4 KPI cards ───────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="kpi green">'
            f'<div class="kpi-lbl">Historical Phase Completion Rates</div>'
            f'<div style="font-size:0.92rem;font-weight:600;color:{THEME["text"]};'
            f'margin:6px 0 4px 0;line-height:1.7">'
            f'Phase 1 &#8594; {hp.get("Phase 1",0):.0f}%<br>'
            f'Phase 2 &#8594; {hp.get("Phase 2",0):.0f}%<br>'
            f'Phase 3 &#8594; {hp.get("Phase 3",0):.0f}%</div>'
            f'<div class="kpi-sub slate" style="font-size:0.7rem">'
            f'Based on historical trial status outcomes<br>'
            f'(Completed vs Terminated / Withdrawn)</div>'
            f'</div>',
            unsafe_allow_html=True)

    with c2:
        # v8: "320 Active Trials Selected Sponsor"
        st.markdown(
            f'<div class="kpi blue">'
            f'<div class="kpi-val">{len(ar)}</div>'
            f'<div class="kpi-lbl">Active Trials</div>'
            f'<div class="kpi-sub slate">Selected Sponsor: {data["DEPLOY_SPONSOR"]}</div>'
            f'</div>',
            unsafe_allow_html=True)

    with c3:
        st.markdown(kpi(str(crit),'Critical Risk Trials',
                        f'{crit} of {len(ar)} active trials — Require Immediate Review','red'),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(kpi(str(p50),'Expected Approvals',
                        '5-Year Base Case (P50)','green'),
                    unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec-hdr">Business Problem</div>', unsafe_allow_html=True)
        st.markdown('''<div class="card">
            <div class="card-title">Why This System Exists</div>
            <div class="card-text">
                Clinical trial termination represents a significant source of portfolio risk
                in pharmaceutical development. Terminated studies consume clinical, operational,
                and financial resources while reducing the probability of future approvals.<br><br>
                This platform helps portfolio managers identify active studies with elevated
                termination risk, prioritize intervention efforts, and evaluate the downstream
                impact of those risks through portfolio-level forecasting and scenario analysis.
            </div>
        </div>''', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">ML Pipeline</div>', unsafe_allow_html=True)
        steps = ['Data Loading','Cleaning','Feature Engineering',
                 'XGBoost Training','Zone Analysis','SHAP Explainability',
                 'Risk Scoring','Monte Carlo Forecast']
        html_steps = ''.join([
            f'<span class="step">{i+1}. {s}</span>'
            + ('<span style="color:#94a3b8;font-size:1rem;padding:0 2px">&#8594;</span>'
               if i < len(steps)-1 else '')
            for i, s in enumerate(steps)])
        st.markdown(f'<div class="step-row">{html_steps}</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec-hdr">Key Deliverables</div>', unsafe_allow_html=True)
        st.markdown(f'''<div class="card">
        <div class="card-title">Risk Scoring</div>
        <div class="card-text">Every active trial receives a risk score between 0 and 1.
        Trials exceeding the critical threshold are prioritized for review and intervention.</div>
        <br>
        <div class="card-title">Risk Escalation Framework</div>
        <div class="card-text">
            <strong>0.30</strong> — Screening Threshold &nbsp;|&nbsp;
            <strong>0.75</strong> — Operational Review &nbsp;|&nbsp;
            <strong>0.95</strong> — Executive Escalation<br>
            <span style="font-size:0.78rem;color:{THEME["text_sub"]}">
            Thresholds derived through historical precision-recall analysis.</span>
        </div>
        <br>
        <div class="card-title">Explainable AI</div>
        <div class="card-text">SHAP analysis identifies the primary drivers behind each risk
        prediction, including enrollment size, sponsor performance, phase, and therapeutic area.</div>
        <br>
        <div class="card-title">Portfolio Forecasting</div>
        <div class="card-text">Monte Carlo simulation combines trial-level risk estimates with
        phase-specific progression assumptions to generate P10, P50, and P90 portfolio outcome
        scenarios over a 5-year horizon.</div>
        <br>
        <div class="card-title">Executive Decision Support</div>
        <div class="card-text">Interactive dashboards provide portfolio health monitoring,
        critical-risk trial identification, therapeutic-area risk analysis, and long-term
        scenario planning.</div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec-hdr">Dataset</div>', unsafe_allow_html=True)
        rows = [('Raw Trials','13,748'),('Training Window','2005-2017'),
                ('Training Trials','9,541'),('Unique Sponsors','10'),
                ('Therapeutic Areas','8'),('Target Variable','Trial Completion')]
        html = ''.join([
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
            f'border-bottom:1px solid {THEME["divider"]};font-size:0.8rem">'
            f'<span style="color:{THEME["text_sub"]}">{k}</span>'
            f'<span style="font-weight:600;color:{THEME["text"]}">{v}</span></div>'
            for k,v in rows])
        st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="sec-hdr">Model Performance</div>', unsafe_allow_html=True)
        rows = [('Algorithm','XGBoost Classifier'),
                ('Validation','Rolling Window CV (4 windows)'),
                ('Rolling Mean AUC',f'{data["mean_auc"]:.3f} +/- {data["std_auc"]:.3f}'),
                ('Test Set AUC',f'{data["xgb_final_auc"]:.3f}'),
                ('Features Used','14'),
                ('Training Trials','8,150')]
        html = ''.join([
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
            f'border-bottom:1px solid {THEME["divider"]};font-size:0.8rem">'
            f'<span style="color:{THEME["text_sub"]}">{k}</span>'
            f'<span style="font-weight:600;color:{THEME["text"]}">{v}</span></div>'
            for k,v in rows])
        st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — EXECUTIVE PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════
def page_executive(data):
    ar   = data['active_ready']
    p10, p50, p90 = data['p10'], data['p50'], data['p90']
    crit = int((ar['risk_label']=='Critical Risk').sum()) if len(ar) else 0
    high = int((ar['risk_label']=='High Risk').sum()) if len(ar) else 0
    total= len(ar)

    st.markdown(
        '<div class="pg-title">Executive Portfolio Overview</div>'
        '<div class="pg-sub">Integrated view of portfolio risk, trial performance, and forecasted pipeline outcomes.</div>',
        unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:1px solid #E5E7EB;margin:4px 0 14px 0">', unsafe_allow_html=True)

    # ── 3 KPIs (removed Predicted Completion Rate and P10-P90 Forecast) ──
    st.markdown('<div class="sec-hdr">Portfolio Risk Indicators</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        at_risk = crit + high
        st.markdown(kpi(at_risk,'Actionable Risk Trials',
                        f'{crit} Critical | {high} High','red'), unsafe_allow_html=True)

    with c2:
        wp  = data['worst_phase']
        wpv = data['pred_phase'].get(wp, 0)
        st.markdown(kpi(wp,'Highest Risk Phase',
                        f'{wpv:.0f}% predicted completion','red'), unsafe_allow_html=True)

    with c3:
        st.markdown(kpi(str(p50),'Expected Approvals',
                        '5-Year Base Case (P50)','green'), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Risk Distribution + Action Panel ──────────────────────
    col_l, col_r = st.columns([1.4, 1])

    with col_l:
        st.markdown('<div class="sec-hdr">Portfolio Risk Distribution</div>', unsafe_allow_html=True)
        if len(ar):
            rc    = ar['risk_label'].value_counts()
            order = ['Critical Risk','High Risk','Moderate Risk','Low Risk']
            labels= [l for l in order if l in rc.index]
            values= [rc[l] for l in labels]
            # v8: show count + percentage
            pcts  = [f'{rc[l]} ({rc[l]/total*100:.0f}%)' for l in labels]
            cmap  = {'Critical Risk':'#C0392B','High Risk':'#B7800A',
                     'Moderate Risk':'#2E6DA4','Low Risk':'#1A6B3A'}
            fig = go.Figure(go.Bar(
                y=labels, x=values, orientation='h',
                marker_color=[cmap[l] for l in labels],
                text=pcts, textposition='outside',
                textfont=dict(color=DARK, size=12, family='Inter')))
            fig = pgo(fig, height=230)
            fig.update_layout(
                # v8: dark color for x axis label
                xaxis=dict(title='Number of Trials',
                           title_font=dict(color=DARK, size=12),
                           tickfont=dict(color=DARK, size=11)),
                yaxis=dict(categoryorder='array', categoryarray=order[::-1],
                           tickfont=dict(color=DARK, size=11)))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec-hdr">Executive Action Panel</div>', unsafe_allow_html=True)
        crit_df = ar[ar['risk_label']=='Critical Risk'] if len(ar) else pd.DataFrame()
        n_p2, n_onc = 0, 0
        top_phase_label = 'None'
        if len(crit_df):
            top_phases = crit_df['Phase'].value_counts()
            n_p2 = int(crit_df[crit_df['Phase']=='Phase 2'].shape[0])
            top_ta = crit_df['therapeutic_area'].value_counts()
            # v8: count ALL in Oncology + related, not hardcode 10
            onc_like = [t for t in top_ta.index
                        if any(w in t.lower() for w in ['oncology','cancer','tumor'])]
            other_like = [t for t in top_ta.index if 'other' in t.lower()]
            onc_count  = int(sum(top_ta.get(t,0) for t in onc_like))
            other_count= int(sum(top_ta.get(t,0) for t in other_like))
            n_onc = onc_count + other_count
            top_phase_label = top_phases.index[0] if len(top_phases) else 'None'

        st.markdown(f'''
        <div class="action-panel">
            <div class="action-title">ACTION REQUIRED</div>
            <div class="action-item">
                <strong style="color:#93c5fd">{crit} Critical Risk Trials</strong>
                Require Review
            </div>
            <div class="action-item" style="margin-top:8px">
                &bull; {n_p2} located in Phase 2 programs
            </div>
            <div class="action-item">
                &bull; {n_onc} located in Oncology and related indications
            </div>
            <div class="action-item">
                &bull; {top_phase_label} represents the largest concentration
                of critical-risk studies
            </div>
            <div class="action-item" style="margin-top:10px;color:#86efac;font-weight:600">
                Recommended Action:
            </div>
            <div class="action-item" style="color:#bbf7d0">
                Prioritize {top_phase_label} portfolio review
                and mitigation planning.
            </div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── SHAP + TA Table side by side ──────────────────────────
    col_shap, col_ta = st.columns(2)

    with col_shap:
        st.markdown('<div class="sec-hdr">Top Predictive Features (SHAP)</div>', unsafe_allow_html=True)
        shap_df = data['shap_df']
        if len(shap_df):
            top5 = shap_df.head(5).copy()
            top5['label'] = top5['feature'].map(lambda x: LABEL_MAP.get(x, x))
            # v8: round to 2 decimal places
            top5['imp2'] = top5['importance'].round(2)
            fig = go.Figure(go.Bar(
                y=top5['label'][::-1],
                x=top5['importance'][::-1],
                orientation='h',
                marker_color=THEME['accent'],
                text=top5['imp2'][::-1],
                textposition='outside',
                textfont=dict(color=DARK, size=11)))
            fig = pgo(fig, height=260)
            fig.update_layout(
                xaxis=dict(
                    title='Mean |SHAP| Value',
                    title_font=dict(color=DARK, size=12),
                    tickfont=dict(color=DARK, size=10)),
                yaxis=dict(tickfont=dict(color=DARK, size=10)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('SHAP values not available.')

    with col_ta:
        st.markdown('<div class="sec-hdr">Therapeutic Area Risk Profile</div>',
                    unsafe_allow_html=True)
        if len(ar):
            ta_total = ar.groupby('therapeutic_area').size().reset_index(name='Total Trials')
            ta_hc    = (ar[ar['risk_label'].isin(['Critical Risk','High Risk'])]
                        .groupby('therapeutic_area').size()
                        .reset_index(name='High + Critical'))
            ta_merged = ta_total.merge(ta_hc, on='therapeutic_area', how='left').fillna(0)
            ta_merged['High + Critical'] = ta_merged['High + Critical'].astype(int)
            ta_merged['Risk Rate'] = (
                ta_merged['High + Critical'] / ta_merged['Total Trials'] * 100
            ).round(1).astype(str) + '%'
            ta_merged = ta_merged.sort_values('High + Critical', ascending=False)
            ta_merged = ta_merged.rename(columns={'therapeutic_area':'Therapeutic Area'})
            st.dataframe(
                ta_merged[['Therapeutic Area','Total Trials','High + Critical','Risk Rate']],
                hide_index=True, use_container_width=True,
                height=min(280, 45 + len(ta_merged)*38))


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — ACTIVE TRIAL MONITOR
# ═══════════════════════════════════════════════════════════════
def page_monitor(filtered, active_ready, shap_df, FEATURE_COLS):
    total  = len(active_ready)
    f_crit = int((active_ready['risk_label']=='Critical Risk').sum())
    f_high = int((active_ready['risk_label']=='High Risk').sum())
    f_mod  = int((active_ready['risk_label']=='Moderate Risk').sum())
    f_low  = int((active_ready['risk_label']=='Low Risk').sum())

    st.markdown(
        '<div class="pg-title">Active Trial Monitor</div>'
        '<div class="pg-sub">Trial-level risk monitoring and prioritization across the active portfolio.</div>',
        unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:1px solid #E5E7EB;margin:4px 0 14px 0">', unsafe_allow_html=True)

    # v8: 4 KPIs — added Watch List (Moderate Risk)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # v8: "Current Portfolio" under Active Trials
        st.markdown(
            f'<div class="kpi blue">'
            f'<div class="kpi-val">{total}</div>'
            f'<div class="kpi-lbl">Active Trials</div>'
            f'<div class="kpi-sub slate">Current Portfolio</div>'
            f'</div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(kpi(f_crit+f_high,'Elevated Risk Trials',
                        f'{f_crit} Critical | {f_high} High','red'), unsafe_allow_html=True)
    with c3:
        # v8: new Watch List KPI
        st.markdown(kpi(f_mod,'Watch List Trials','Moderate Risk','orange'), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi(f_low,'Low Risk Trials','Routine Oversight','green'), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Critical Risk Rate by Phase (Phase 1/2/3 only) ────────
    st.markdown('<div class="sec-hdr">Critical Risk Rate by Phase</div>', unsafe_allow_html=True)

    # v8: only Phase 1, Phase 2, Phase 3
    phase_data = []
    for ph in ['Phase 1','Phase 2','Phase 3']:
        ph_df   = active_ready[active_ready['Phase'] == ph]
        ph_crit = active_ready[
            (active_ready['Phase']==ph) &
            (active_ready['risk_label']=='Critical Risk')]
        if len(ph_df) > 0:
            phase_data.append({
                'Phase': ph,
                'Critical Trials': len(ph_crit),
                'Total Trials'   : len(ph_df),
                'Critical Risk Rate': f'{len(ph_crit)/len(ph_df)*100:.1f}%'
            })
    if phase_data:
        phase_tbl = pd.DataFrame(phase_data)
        st.dataframe(phase_tbl, hide_index=True, use_container_width=True,
                     height=min(200, 45 + len(phase_tbl)*38))

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Critical Risk Table ────────────────────────────────────
    st.markdown('<div class="sec-hdr">Critical Risk Trials — Priority Review List</div>',
                unsafe_allow_html=True)

    src = filtered if len(filtered) else active_ready
    crit_tbl = src[src['risk_label']=='Critical Risk'].copy()
    crit_tbl  = crit_tbl.sort_values('risk_score', ascending=False)

    if len(crit_tbl) == 0:
        st.success('No Critical Risk trials in current selection.')
    else:
        if 'Enrollment' in crit_tbl.columns:
            crit_tbl['priority_score'] = (
                crit_tbl['risk_score'] *
                np.log1p(crit_tbl['Enrollment'].fillna(0))).round(2)
        else:
            crit_tbl['priority_score'] = crit_tbl['risk_score'].round(3)

        display_cols = ['NCT','Phase','Condition','therapeutic_area',
                        'Enrollment','risk_label','risk_score',
                        'success_probability','priority_score']
        display_cols = [c for c in display_cols if c in crit_tbl.columns]
        show = crit_tbl[display_cols].head(30).copy()
        show.rename(columns={
            'therapeutic_area'   : 'Area',
            'risk_label'         : 'Risk Tier',
            'risk_score'         : 'Risk Score',
            'success_probability': 'Predicted Completion',   # v8: renamed
            'priority_score'     : 'Priority Score'
        }, inplace=True)
        if 'Risk Score' in show.columns:
            show['Risk Score'] = show['Risk Score'].round(3)
        if 'Predicted Completion' in show.columns:
            show['Predicted Completion'] = (
                show['Predicted Completion']*100).round(1).astype(str)+'%'

        st.dataframe(show, hide_index=True, use_container_width=True,
                     height=min(420, 45+len(show)*36))
        csv = crit_tbl[display_cols].to_csv(index=False)
        st.download_button('Download Critical Risk List (CSV)', data=csv,
                           file_name='critical_risk_trials.csv', mime='text/csv')

    st.markdown('<br>', unsafe_allow_html=True)

    # ── TA Risk Concentration ──────────────────────────────────
    st.markdown('<div class="sec-hdr">Risk Concentration by Therapeutic Area</div>',
                unsafe_allow_html=True)
    if len(active_ready):
        ta_risk = (active_ready.groupby(['therapeutic_area','risk_label'])
                   .size().reset_index(name='count'))
        fig = px.bar(ta_risk, x='therapeutic_area', y='count', color='risk_label',
                     color_discrete_map={
                         'Critical Risk':'#C0392B','High Risk':'#B7800A',
                         'Moderate Risk':'#2E6DA4','Low Risk':'#1A6B3A'},
                     category_orders={'risk_label':['Critical Risk','High Risk',
                                                    'Moderate Risk','Low Risk']})
        fig = pgo(fig, height=300, legend=True)
        fig.update_layout(
            barmode='stack', xaxis_title='',
            # v8: dark y axis label
            yaxis=dict(title='Trial Count',
                       title_font=dict(color=DARK, size=12),
                       tickfont=dict(color=DARK, size=11)),
            xaxis=dict(tickangle=-30, tickfont=dict(color=DARK, size=10)),
            legend=dict(orientation='h', y=1.1, font=dict(color=DARK, size=10)))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — PORTFOLIO FORECAST & SCENARIO PLANNING
# ═══════════════════════════════════════════════════════════════
def page_forecast(data):
    scenarios  = data['scenarios']
    phases_df  = data['phases_df']
    approvals  = data['approvals']
    active_ready = data['active_ready']
    p10, p50, p90 = data['p10'], data['p50'], data['p90']

    st.markdown(
        '<div class="pg-title">Portfolio Forecast & Scenario Planning</div>'
        '<div class="pg-sub">Monte Carlo simulation — 10,000 runs — 5-year estimated approval forecast</div>',
        unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:1px solid #E5E7EB;margin:4px 0 14px 0">', unsafe_allow_html=True)

    if scenarios.empty or phases_df.empty:
        st.warning('Forecast data not available. Run main.py first.')
        return

    # ── 3 Scenario KPIs ───────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(kpi(p10,'Pessimistic Scenario','P10 Forecast','red'), unsafe_allow_html=True)
    with c2:
        # v8: "P50 Forecast" under Base Case Scenario
        st.markdown(
            f'<div class="kpi blue">'
            f'<div class="kpi-val">{p50}</div>'
            f'<div class="kpi-lbl">Base Case Scenario</div>'
            f'<div class="kpi-sub slate">P50 Forecast</div>'
            f'</div>',
            unsafe_allow_html=True)
    with c3: st.markdown(kpi(p90,'Optimistic Scenario','P90 Forecast','green'), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Year-by-Year chart ────────────────────────────────────
    st.markdown('<div class="sec-hdr">Expected Annual Approvals by Scenario</div>',
                unsafe_allow_html=True)

    years  = [f'Year {int(r["year"])}' for _, r in phases_df.iterrows()]
    p10_yr = phases_df['p10'].tolist()
    p50_yr = phases_df['p50'].tolist()
    p90_yr = phases_df['p90'].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years+years[::-1], y=p90_yr+p10_yr[::-1],
        fill='toself', fillcolor='rgba(46,109,164,0.18)',
        line=dict(color='rgba(0,0,0,0)'), name='Forecast Uncertainty Band'))
    fig.add_trace(go.Bar(
        x=years, y=p50_yr, name='Base Case (P50)',
        marker_color=THEME['accent'], opacity=0.9,
        text=p50_yr, textposition='outside',
        textfont=dict(color=DARK, size=14, family='Inter')))
    fig.add_trace(go.Scatter(
        x=years, y=p10_yr, mode='lines+markers', name='Pessimistic (P10)',
        line=dict(color=THEME['danger'], dash='dash', width=2),
        marker=dict(size=7, color=THEME['danger'])))
    fig.add_trace(go.Scatter(
        x=years, y=p90_yr, mode='lines+markers', name='Optimistic (P90)',
        line=dict(color=THEME['positive'], dash='dash', width=2),
        marker=dict(size=7, color=THEME['positive'])))

    fig = pgo(fig, height=360, legend=True, margin=dict(t=20,b=20,l=8,r=8))
    fig.update_layout(
        # v8: dark y axis label
        yaxis=dict(title='Estimated Approvals', rangemode='tozero',
                   title_font=dict(color=DARK, size=12),
                   tickfont=dict(color=DARK, size=11)),
        xaxis=dict(tickfont=dict(color=DARK, size=11)),
        legend=dict(orientation='h', y=1.07, x=0,
                    font=dict(color=DARK, size=11)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="note-blue">'
        f'Base Case (P50) forecasts <strong>{p50} approvals</strong> over five years, '
        f'with outcomes ranging from <strong>{p10} (P10)</strong> to '
        f'<strong>{p90} (P90)</strong>.'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Waterfall + Distribution ───────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec-hdr">Drivers of Expected Approvals (P50)</div>',
                    unsafe_allow_html=True)

        if len(active_ready):
            from src.config import TIME_FRACTION, PHASE_CAPS, PHASE_MAP as PM
            sim_data = active_ready.copy()
            sim_data['phase_num_s'] = sim_data['Phase'].map(PM).fillna(0).astype(int)
            sim_data = sim_data[sim_data['phase_num_s'].isin([1,2,3])]

            contribs = {}
            for pn, ph in [(3,'Phase 3'),(2,'Phase 2'),(1,'Phase 1')]:
                g = sim_data[sim_data['phase_num_s']==pn]
                if len(g):
                    avg = g['success_probability'].mean()
                    # v8: round to 1 decimal to avoid long floats
                    contribs[ph] = round(
                        len(g) * min(avg, PHASE_CAPS[pn]) * TIME_FRACTION[pn], 1)
                else:
                    contribs[ph] = 0.0

            total_c = sum(contribs.values())
            colors_wf = {'Phase 3':THEME['accent'],'Phase 2':THEME['warning'],'Phase 1':'#64748b'}

            for ph_label, val in contribs.items():
                pct   = round(val/max(total_c,1)*100)
                bar_w = max(int(pct), 2)
                color = colors_wf[ph_label]
                # v8: label "Phase X Expected Approvals" + rounded value
                display_val = f'{val:.1f}'
                st.markdown(
                    f'<div class="wf-bar">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<span class="wf-label">{ph_label} Expected Approvals</span>'
                    f'<span class="wf-val" style="color:{color}">{display_val} ({pct}%)</span></div>'
                    f'<div style="background:{THEME["divider"]};border-radius:3px;'
                    f'height:6px;margin-top:6px">'
                    f'<div style="background:{color};width:{bar_w}%;height:100%;'
                    f'border-radius:3px"></div></div></div>',
                    unsafe_allow_html=True)

            st.markdown(
                f'<div style="padding:10px 16px;background:{THEME["header"]};'
                f'color:white;border-radius:6px;margin-top:4px;'
                f'display:flex;justify-content:space-between">'
                f'<span style="font-weight:600;font-size:0.85rem">P50 Total</span>'
                f'<span style="font-weight:700;font-size:1.1rem">{p50}</span></div>',
                unsafe_allow_html=True)

            p3_pct = round(contribs.get('Phase 3',0)/max(total_c,1)*100)
            st.markdown(
                f'<div class="note-blue" style="margin-top:8px">'
                f'Phase 3 programs contribute <strong>{p3_pct}%</strong> of expected approvals '
                f'under the base-case forecast and represent the primary driver of '
                f'portfolio outcomes.</div>',
                unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec-hdr">Simulation Distribution (10,000 Runs)</div>',
                    unsafe_allow_html=True)
        if not approvals.empty:
            fig3 = px.histogram(approvals, x='total_approvals', nbins=28,
                                color_discrete_sequence=[THEME['accent']], opacity=0.8)
            for xv, col_v, lbl in [
                (p10, THEME['danger'],   f'P10={p10}'),
                (p50, THEME['accent'],   f'P50={p50}'),
                (p90, THEME['positive'], f'P90={p90}')]:
                fig3.add_vline(x=xv, line_dash='dash', line_color=col_v,
                               annotation_text=lbl, annotation_position='top right',
                               annotation_font_color=col_v, annotation_font_size=11)
            fig3 = pgo(fig3, height=300)
            fig3.update_layout(
                # v8: dark axis labels, corrected axis titles
                xaxis=dict(title='Total Approvals Over 5 Years',
                           title_font=dict(color=DARK, size=12),
                           tickfont=dict(color=DARK, size=11)),
                yaxis=dict(title='Simulation Count',
                           title_font=dict(color=DARK, size=12),
                           tickfont=dict(color=DARK, size=11)),
                showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    # v8: Forecast Methodology note (replaces Important note + Stakeholder Guidance)
    st.markdown('''
    <div class="note-amber">
        <strong>Forecast Methodology</strong><br>
        The predictive model estimates the probability of trial completion rather than
        regulatory approval. Portfolio forecasts combine completion probabilities with
        phase-based approval assumptions and should be interpreted as scenario-planning
        estimates, not direct approval predictions.
    </div>''', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────
def main():
    with st.spinner('Initialising pipeline — first load takes ~90 seconds with SHAP...'):
        data = run_pipeline()

    page, filtered = render_sidebar(data['active_ready'])

    if page == 'Home':
        page_home(data)
    elif page == 'Executive Portfolio Overview':
        page_executive(data)
    elif page == 'Active Trial Monitor':
        page_monitor(filtered, data['active_ready'],
                     data['shap_df'], data['FEATURE_COLS'])
    elif page == 'Portfolio Forecast & Scenario Planning':
        page_forecast(data)


if __name__ == '__main__':
    main()
