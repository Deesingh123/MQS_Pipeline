import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#DB_FILE = os.path.join(BASE_DIR, "mqs.db")

st.set_page_config(page_title="MQS Yield Intelligence", layout="wide", page_icon="📊",
                   initial_sidebar_state="expanded")

BLUE="#2563EB"; GREEN="#22c55e"; RED="#ef4444"; ORANGE="#f97316"
PURPLE="#8b5cf6"; TEAL="#14b8a6"; YELLOW="#eab308"; BORDER="#E2E8F0"

def rgba(h, a=0.12):
    h=h.lstrip('#'); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f'rgba({r},{g},{b},{a})'

st.markdown(f"""
<style>
  .stApp{{background:#FAFBFF}}
  .block-container{{padding-top:.5rem;padding-bottom:.5rem}}
  div[data-testid="stMetric"]{{
    background:#fff;border:1px solid {BORDER};border-radius:14px;
    padding:12px 16px;box-shadow:0 2px 8px rgba(37,99,235,.07)}}
  div[data-testid="stMetric"] label{{color:{BLUE}!important;font-size:.78rem;font-weight:600}}
  div[data-testid="stMetricValue"]{{font-size:1.6rem!important;font-weight:700;color:#0f172a}}
  div[data-testid="stMetricDelta"]{{font-size:.8rem!important}}
  h1,h2,h3{{color:{BLUE}}}
  .stTabs [data-baseweb="tab"]{{border-radius:8px 8px 0 0;padding:6px 16px;background:#f1f5f9}}
  .stTabs [aria-selected="true"]{{background:{BLUE}15;border-bottom:3px solid {BLUE}}}
  .sh{{background:linear-gradient(90deg,{BLUE}22,transparent);
    border-left:4px solid {BLUE};padding:6px 14px;border-radius:0 8px 8px 0;
    margin:.6rem 0;font-weight:700;color:{BLUE};font-size:1rem}}
  .sh-r{{background:linear-gradient(90deg,{RED}22,transparent);
    border-left:4px solid {RED};padding:6px 14px;border-radius:0 8px 8px 0;
    margin:.6rem 0;font-weight:700;color:{RED};font-size:1rem}}
  .sh-g{{background:linear-gradient(90deg,{GREEN}22,transparent);
    border-left:4px solid {GREEN};padding:6px 14px;border-radius:0 8px 8px 0;
    margin:.6rem 0;font-weight:700;color:#15803d;font-size:1rem}}
</style>""", unsafe_allow_html=True)

# ── DB LOAD ──────────────────────────────────────────────────────────────────
def get_url():
    return st.secrets["DATABASE_URL"]

@st.cache_data(ttl=90)
def load_data():
    try:
        conn = psycopg2.connect(get_url(), connect_timeout=15)
        df = pd.read_sql('SELECT * FROM rld1', conn)
        hdf = pd.read_sql('SELECT * FROM rld1_hidden', conn)
        conn.close(); return df, hdf
    except Exception as e:
        st.error(f"DB error: {e}"); return pd.DataFrame(), pd.DataFrame()

# ── NUMERIC CAST ──────────────────────────────────────────────────────────────
NUM_COLS = ['Prime_Pass','Prime_Fail','Prime_Handle','TotPass','TotFail','TotHandle',
            'PrimeCount','TotalNTFCount','PrimeDefectCount','TotalDefect',
            'PrimeDPHU','TotalDPHU','PMYield','TotalYield','TimePeriod']

def cast_num(df):
    d = df.copy()
    for c in NUM_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0)
    return d

# ── AGGREGATE KPIs (CORRECT — sum first, then divide) ──────────────────────
def agg_kpis(df):
    """Compute KPIs by summing numerators & denominators, NOT by averaging rates."""
    PP  = df['Prime_Pass'].sum()
    PH  = df['Prime_Handle'].sum()
    NTF = df['PrimeCount'].sum()
    PF  = df['Prime_Fail'].sum()
    TH  = df['TotHandle'].sum()
    TD  = df['TotalDefect'].sum()
    TF  = df['TotFail'].sum()
    TNTF= df['TotalNTFCount'].sum()

    fty  = round(PP / PH * 100, 4) if PH > 0 else 0.0
    fpy  = round((PP + NTF) / PH * 100, 4) if PH > 0 else 0.0
    dphu = round(TD / TH * 100, 4) if TH > 0 else 0.0
    fail = round(PF / PH * 100, 4) if PH > 0 else 0.0
    ntfp = round(NTF / PH * 100, 4) if PH > 0 else 0.0
    gap  = round(fpy - fty, 4)
    rec  = round((fty - fpy) / (100 - fpy) * 100, 4) if 0 < fpy < 99.5 else 0.0
    return dict(PP=PP, PH=int(PH), NTF=int(NTF), PF=int(PF), TH=int(TH),
                TD=int(TD), TF=int(TF), TNTF=int(TNTF),
                FTY=fty, FPY=fpy, DPHU=dphu, Fail=fail,
                NTFPct=ntfp, Gap=gap, Rec=rec)

# ── RTY — correct: aggregate per process FIRST, then multiply ───────────────
def compute_rty(df):
    """RTY = ∏(per-process FPY) with per-process FPY = sum(PP+NTF)/sum(PH) per process."""
    rows = []
    for proc, g in df.groupby('Process'):
        ph = g['Prime_Handle'].sum()
        if ph <= 0:
            continue
        pp  = g['Prime_Pass'].sum()
        ntf = g['PrimeCount'].sum()
        fpy = (pp + ntf) / ph  # fraction
        if fpy <= 0:
            continue
        rows.append({'Process': proc, 'FPY_pct': round(fpy * 100, 4), 'FPY_frac': fpy})
    if not rows:
        return pd.DataFrame(), 0.0
    rdf = pd.DataFrame(rows).sort_values('FPY_pct', ascending=False)
    rty_val = round(float(np.prod(rdf['FPY_frac'])) * 100, 4)
    rdf['RTY_running'] = (rdf['FPY_frac'].cumprod() * 100).round(4)
    return rdf, rty_val

def compute_rolled_fty_overall(df):
    """Overall Rolled FTY = ∏(per-process FTY_i) where FTY_i = Σ(Prime_Pass)/Σ(Prime_Handle) per process."""
    fracs = []
    for proc, g in df.groupby('Process'):
        ph = g['Prime_Handle'].sum(); pp = g['Prime_Pass'].sum()
        if ph > 0: fracs.append(pp / ph)
    if not fracs: return 0.0
    import math
    return round(math.prod(fracs) * 100, 2)


# ── ALTERNATIVE FTY/FPY INSIGHT (Image-3 formula) ───────────────────────────
# FTY_alt = Prime_Pass / max(Prime_Handle across all processes in group)
# FPY_alt = (Prime_Pass + NTF) / Prime_Handle  [per process, same as normal FPY]
def alt_yield_by_process(df):
    """Per image-3: FTY uses the highest prime handle as common denominator."""
    max_ph = df['Prime_Handle'].sum()   # or per-line max; use overall max
    rows = []
    for proc, g in df.groupby('Process'):
        ph  = g['Prime_Handle'].sum()
        pp  = g['Prime_Pass'].sum()
        ntf = g['PrimeCount'].sum()
        pf  = g['Prime_Fail'].sum()
        fpy_alt = round((pp + ntf) / ph * 100, 4) if ph > 0 else np.nan
        fty_alt = round(pp / max_ph * 100, 4) if max_ph > 0 else np.nan
        rows.append({'Process': proc, 'Prime_Handle': int(ph),
                     'Prime_Pass': int(pp), 'NTF': int(ntf),
                     'FPY_alt': fpy_alt, 'FTY_alt': fty_alt,
                     'FTY_std': round(pp/ph*100,4) if ph>0 else np.nan})
    return pd.DataFrame(rows)

def alt_yield_by_line(df):
    """Per image-3 logic applied to Line."""
    max_ph = df['Prime_Handle'].sum()
    rows = []
    for line, g in df.groupby('Line'):
        ph  = g['Prime_Handle'].sum()
        pp  = g['Prime_Pass'].sum()
        ntf = g['PrimeCount'].sum()
        fpy_alt = round((pp + ntf) / ph * 100, 4) if ph > 0 else np.nan
        fty_alt = round(pp / max_ph * 100, 4) if max_ph > 0 else np.nan
        rows.append({'Line': line, 'Prime_Handle': int(ph),
                     'Prime_Pass': int(pp), 'NTF': int(ntf),
                     'FPY_alt': fpy_alt, 'FTY_alt': fty_alt,
                     'FTY_std': round(pp/ph*100,4) if ph>0 else np.nan})
    return pd.DataFrame(rows)

# ── GAUGE ─────────────────────────────────────────────────────────────────────
def gauge_fig(value, title, lo=85, mid=95, invert=False):
    if invert:
        col = GREEN if value >= mid else (ORANGE if value >= lo else RED)
    else:
        col = GREEN if value >= mid else (ORANGE if value >= lo else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 2),
        title={'text': title, 'font': {'size': 12}},
        number={'suffix': '%', 'font': {'size': 18, 'color': col}},
        gauge={'axis': {'range': [0, 100], 'tickwidth': 1},
               'bar': {'color': col, 'thickness': .28},
               'bgcolor': '#f1f5f9',
               'steps': [{'range': [0, lo], 'color': '#fecaca'},
                          {'range': [lo, mid], 'color': '#fef3c7'},
                          {'range': [mid, 100], 'color': '#d1fae5'}],
               'threshold': {'line': {'color': '#1e3a8a', 'width': 3},
                              'thickness': .8, 'value': mid}}))
    fig.update_layout(height=200, margin=dict(l=18, r=18, t=42, b=4),
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig

# ── PARETO ────────────────────────────────────────────────────────────────────
def pareto_fig(df, label_col, value_col, title):
    d = df[df[value_col] > 0].sort_values(value_col, ascending=False).head(20).copy()
    if d.empty:
        return go.Figure()
    d['cumpct'] = d[value_col].cumsum() / d[value_col].sum() * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=d[label_col], y=d[value_col], name='Count',
        marker_color=BLUE, opacity=.85,
        text=d[value_col].round(2).astype(str), textposition='outside'), secondary_y=False)
    fig.add_trace(go.Scatter(x=d[label_col], y=d['cumpct'],
        mode='lines+markers', name='Cumulative %',
        line=dict(color=RED, width=2.5), marker=dict(size=7)), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color=ORANGE, line_width=2,
        annotation_text="80% (Pareto)", annotation_font=dict(color=ORANGE, size=12),
        secondary_y=True)
    fig.update_yaxes(title_text="Count", range=[0, d[value_col].max()*1.2], secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105],
        tickvals=[0,20,40,60,80,100], secondary_y=True)
    fig.update_layout(title=title, height=370, margin=dict(l=40,r=50,t=55,b=80),
        xaxis_tickangle=-40, legend=dict(orientation="h",yanchor="bottom",y=1.02))
    return fig

# ═══════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════
df_raw, hdf_raw = load_data()
if df_raw.empty:
    st.warning("⚠️ No data in mqs_yield.db. Run the scraper first."); st.stop()

df_all = cast_num(df_raw)
hdf_all = hdf_raw.copy()
for c in ['PFail','TFail','PDef','TDef','PHandle','THandle','PDPHU','TDPHU']:
    if c in hdf_all.columns:
        hdf_all[c] = pd.to_numeric(hdf_all[c], errors='coerce').fillna(0)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔽 Filters")
    dates = sorted(df_all['Date'].dropna().unique(), reverse=True)
    sel_dates = st.multiselect("📅 Date(s)", dates, default=dates[:3] if dates else [])
    all_slots = sorted(df_all['SlotLabel'].dropna().unique())
    sel_slots = st.multiselect("⏱ Slot(s)", all_slots, default=all_slots)
    all_lines = sorted(df_all['Line'].dropna().unique())
    sel_lines = st.multiselect("🏭 Line(s)", all_lines, default=all_lines)
    all_proc  = sorted(df_all['Process'].dropna().unique())
    sel_proc  = st.multiselect("⚙️ Process", all_proc, default=all_proc)
    all_tech  = sorted(df_all['Technology'].dropna().unique())
    sel_tech  = st.multiselect("🔬 Technology", all_tech, default=all_tech)
    all_fam   = sorted(df_all['Family'].dropna().unique())
    sel_fam   = st.multiselect("👨‍👩‍👧 Family", all_fam, default=all_fam)
    st.markdown("---")
    target_fpy = st.slider("🎯 FPY Target %", 80, 100, 95)
    target_fty = st.slider("🎯 FTY Target %", 80, 100, 97)
    st.markdown("---")
    if st.button("🔄 Refresh"): st.cache_data.clear(); st.rerun()
    st.caption(f"rld1: **{len(df_all):,}** rows")
    st.caption(f"hidden: **{len(hdf_all):,}** rows")

# ── FILTER ────────────────────────────────────────────────────────────────────
def filt(df):
    m = pd.Series([True]*len(df))
    if sel_dates and 'Date' in df.columns: m &= df['Date'].isin(sel_dates)
    if sel_slots and 'SlotLabel' in df.columns: m &= df['SlotLabel'].isin(sel_slots)
    if sel_lines and 'Line' in df.columns:  m &= df['Line'].isin(sel_lines)
    if sel_proc  and 'Process' in df.columns: m &= df['Process'].isin(sel_proc)
    if sel_tech  and 'Technology' in df.columns: m &= df['Technology'].isin(sel_tech)
    if sel_fam   and 'Family' in df.columns: m &= df['Family'].isin(sel_fam)
    return df[m].copy()

df  = filt(df_all)
hf  = hdf_all[hdf_all['Line'].isin(sel_lines)].copy() if sel_lines else hdf_all.copy()
if sel_slots and 'SlotLabel' in hf.columns:
    hf = hf[hf['SlotLabel'].isin(sel_slots)]
if df.empty:
    st.warning("No data matches current filters."); st.stop()

# ── COMPUTE KPIs ──────────────────────────────────────────────────────────────
K = agg_kpis(df)
rty_df, OV_RTY = compute_rty(df)
OV_RFTY = compute_rolled_fty_overall(df)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:.5rem 1rem .6rem;
  background:linear-gradient(135deg,{BLUE}12,{TEAL}08);
  border-bottom:2px solid {BLUE}28;margin-bottom:.8rem">
  <h1 style="font-size:2rem;font-weight:900;letter-spacing:-1px;
    background:linear-gradient(90deg,{BLUE},{TEAL});
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">
    📊 MQS YIELD INTELLIGENCE DASHBOARD
  </h1>
  <div style="color:#64748b;font-size:.85rem;margin-top:.15rem">
    FPY · FTY · RTY · DPHU · NTF · Pareto · Process-Line-Technology Insights
  </div>
</div>""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
c1.metric("🎯 FPY",     f"{K['FPY']:.2f}%")
c2.metric("🔁 FTY",     f"{K['FTY']:.2f}%")
c3.metric("↔ Gap",      f"{K['Gap']:.2f}%")
c4.metric("🔄 RTY",     f"{OV_RTY:.2f}%")
c5.metric("🛡 DPHU",    f"{K['DPHU']:.2f}")
c6.metric("❌ Fail%",   f"{K['Fail']:.2f}%")
c7.metric("🔘 NTF%",    f"{K['NTFPct']:.2f}%")
c8.metric("📦 Handles", f"{K['PH']:,}")

# ── GAUGES ────────────────────────────────────────────────────────────────────
g1,g2,g3,g4,g5 = st.columns(5)
with g1: st.plotly_chart(gauge_fig(K['FPY'],"Overall FPY",target_fpy-10,target_fpy),use_container_width=True)
with g2: st.plotly_chart(gauge_fig(K['FTY'],"Std FTY (Pass/Handle)",target_fty-10,target_fty),use_container_width=True)
with g3: st.plotly_chart(gauge_fig(OV_RFTY,"Rolled FTY (∏Process)",target_fty-10,target_fty),use_container_width=True)
with g4: st.plotly_chart(gauge_fig(OV_RTY,"RTY (Rolled Throughput)",80,90),use_container_width=True)
with g5: st.plotly_chart(gauge_fig(max(0,100-K['DPHU']*10),"DPHU Health (↑=better)",60,85),use_container_width=True)
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
T1,T2,T3,T4,T5,T6,T7 = st.tabs([
    "📈 Yield Trends","⚙️ Process · Line","🔬 Technology · Family",
    "🔄 Alt-FTY/FPY Insight","❌ Defect · NTF · DPHU",
    "📉 Pareto · RTY","🗂 Raw Data"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — YIELD TRENDS
# ════════════════════════════════════════════════════════════════
with T1:
    # Slot trend
    st.markdown('<div class="sh">📈 FPY vs FTY vs Gap — Hourly Slot Trend</div>',unsafe_allow_html=True)
    sl_agg = df.groupby('SlotLabel').apply(lambda g: pd.Series({
        'FPY': round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,4)
               if g['Prime_Handle'].sum()>0 else np.nan,
        'FTY': round(g['Prime_Pass'].sum()/g['Prime_Handle'].sum()*100,4)
               if g['Prime_Handle'].sum()>0 else np.nan,
        'Handles': g['Prime_Handle'].sum(),
    })).reset_index()
    sl_agg['Gap'] = (sl_agg['FPY'] - sl_agg['FTY']).round(4)

    fig_s = make_subplots(specs=[[{"secondary_y":True}]])
    fig_s.add_trace(go.Scatter(x=sl_agg['SlotLabel'],y=sl_agg['FPY'],
        mode='lines+markers',name='FPY',line=dict(color=BLUE,width=2.5),marker=dict(size=7)),secondary_y=False)
    fig_s.add_trace(go.Scatter(x=sl_agg['SlotLabel'],y=sl_agg['FTY'],
        mode='lines+markers',name='FTY',line=dict(color=GREEN,width=2.5,dash='dot'),marker=dict(size=7)),secondary_y=False)
    fig_s.add_trace(go.Bar(x=sl_agg['SlotLabel'],y=sl_agg['Gap'],
        name='Gap',marker_color=ORANGE,opacity=.5),secondary_y=True)
    fig_s.add_hline(y=target_fpy,line_dash="dash",line_color=BLUE,
        annotation_text=f"FPY {target_fpy}%",secondary_y=False)
    fig_s.add_hline(y=target_fty,line_dash="dash",line_color=GREEN,
        annotation_text=f"FTY {target_fty}%",secondary_y=False)
    fig_s.update_yaxes(title_text="Yield %",range=[75,105],secondary_y=False)
    fig_s.update_yaxes(title_text="Gap %",range=[0,10],secondary_y=True)
    fig_s.update_layout(height=360,margin=dict(l=30,r=50,t=20,b=50),
        legend=dict(orientation="h",yanchor="bottom",y=1.02),xaxis_tickangle=-25)
    st.plotly_chart(fig_s,use_container_width=True)

    # ── FPY vs Std FTY vs Rolled FTY — Line comparison ────────────────────────
    st.markdown('<div class="sh">🔀 FPY vs Std FTY vs Rolled FTY — by Line</div>',unsafe_allow_html=True)
    tri = df.groupby('Line').apply(lambda g: pd.Series({
        'FPY':       round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,2)
                     if g['Prime_Handle'].sum()>0 else np.nan,
        'Std_FTY':   round(g['Prime_Pass'].sum()/g['Prime_Handle'].sum()*100,2)
                     if g['Prime_Handle'].sum()>0 else np.nan,
    })).reset_index()
    # Rolled FTY per line
    import math
    rfty_rows=[]
    for ln, lg in df.groupby('Line'):
        fracs=[lg[lg['Process']==p]['Prime_Pass'].sum()/lg[lg['Process']==p]['Prime_Handle'].sum()
               for p in lg['Process'].unique()
               if lg[lg['Process']==p]['Prime_Handle'].sum()>0]
        rfty_rows.append({'Line':ln,'Rolled_FTY':round(math.prod(fracs)*100,2) if fracs else np.nan})
    rfty_df = pd.DataFrame(rfty_rows)
    tri = tri.merge(rfty_df, on='Line', how='left')
    fig_tri = go.Figure()
    fig_tri.add_trace(go.Bar(x=tri['Line'],y=tri['FPY'],name='FPY',marker_color=BLUE,
        text=tri['FPY'].round(2).astype(str)+'%',textposition='outside'))
    fig_tri.add_trace(go.Bar(x=tri['Line'],y=tri['Std_FTY'],name='Std FTY',marker_color=GREEN,
        text=tri['Std_FTY'].round(2).astype(str)+'%',textposition='outside'))
    fig_tri.add_trace(go.Scatter(x=tri['Line'],y=tri['Rolled_FTY'],mode='lines+markers',
        name='Rolled FTY',line=dict(color=PURPLE,width=2.5,dash='dot'),marker=dict(size=9)))
    fig_tri.add_hline(y=target_fpy,line_dash="dash",line_color=RED,annotation_text=f"Target {target_fpy}%")
    fig_tri.update_layout(barmode='group',yaxis=dict(range=[0,112],title='Yield %'),
        height=360,margin=dict(l=20,r=20,t=15,b=40),
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    st.plotly_chart(fig_tri,use_container_width=True)
    # Summary table
    tri['FPY-FTY Gap'] = (tri['FPY']-tri['Std_FTY']).round(2)
    tri['Std-Rolled Gap'] = (tri['Std_FTY']-tri['Rolled_FTY']).round(2)
    st.caption("**FPY vs Std FTY vs Rolled FTY — Summary Table**")
    st.dataframe(tri[['Line','FPY','Std_FTY','Rolled_FTY','FPY-FTY Gap','Std-Rolled Gap']].round(2),
        use_container_width=True, hide_index=True)
    st.info("**FPY > Std FTY**: NTF units inflate FPY. **Std FTY > Rolled FTY**: process chain degradation across the line.")

    # TimePeriod
    if 'TimePeriod' in df.columns:
        st.markdown('<div class="sh">🕐 Hour-of-Day vs FPY / FTY</div>',unsafe_allow_html=True)
        tp = df.groupby('TimePeriod').apply(lambda g: pd.Series({
            'FPY': round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,4)
                   if g['Prime_Handle'].sum()>0 else np.nan,
            'FTY': round(g['Prime_Pass'].sum()/g['Prime_Handle'].sum()*100,4)
                   if g['Prime_Handle'].sum()>0 else np.nan,
        })).reset_index().sort_values('TimePeriod')
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(x=tp['TimePeriod'],y=tp['FPY'],
            mode='lines+markers',name='FPY',line=dict(color=BLUE,width=2.5),
            fill='tozeroy',fillcolor=rgba(BLUE,0.1)))
        fig_tp.add_trace(go.Scatter(x=tp['TimePeriod'],y=tp['FTY'],
            mode='lines+markers',name='FTY',line=dict(color=GREEN,width=2,dash='dot')))
        fig_tp.add_hline(y=target_fpy,line_dash="dash",line_color=RED,
            annotation_text=f"Target {target_fpy}%")
        fig_tp.update_layout(height=290,xaxis_title="Hour of Day",
            yaxis=dict(range=[75,105],title="Yield %"),
            margin=dict(l=30,r=20,t=15,b=40),
            legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig_tp,use_container_width=True)

    # Date×Slot heatmap
    if 'Date' in df.columns:
        st.markdown('<div class="sh">🗓 FPY Heatmap — Date × Slot</div>',unsafe_allow_html=True)
        ds = df.groupby(['Date','SlotLabel']).apply(lambda g: pd.Series({
            'FPY': round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,2)
                   if g['Prime_Handle'].sum()>0 else np.nan
        })).reset_index()
        pv = ds.pivot(index='Date',columns='SlotLabel',values='FPY')
        fig_dh = px.imshow(pv,text_auto='.1f',color_continuous_scale='RdYlGn',
            zmin=80,zmax=100,aspect='auto',labels=dict(color='FPY %'))
        fig_dh.update_layout(height=max(180,len(pv)*46+80),margin=dict(l=10,r=10,t=15,b=15))
        st.plotly_chart(fig_dh,use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — PROCESS · LINE
# ════════════════════════════════════════════════════════════════
with T2:
    # Aggregate correctly per process
    proc_a = df.groupby('Process').apply(lambda g: pd.Series({
        'FPY':    round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,4)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'FTY':    round(g['Prime_Pass'].sum()/g['Prime_Handle'].sum()*100,4)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'FailPct':round(g['Prime_Fail'].sum()/g['Prime_Handle'].sum()*100,4)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'NTFPct': round(g['PrimeCount'].sum()/g['Prime_Handle'].sum()*100,4)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'NTFCount':int(g['PrimeCount'].sum()),
        'Handles': int(g['Prime_Handle'].sum()),
    })).reset_index().dropna(subset=['FPY'])

    line_a = df.groupby('Line').apply(lambda g: pd.Series({
        'FPY':    round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,4)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'FTY':    round(g['Prime_Pass'].sum()/g['Prime_Handle'].sum()*100,4)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'FailPct':round(g['Prime_Fail'].sum()/g['Prime_Handle'].sum()*100,4)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'DPHU':   round(g['TotalDefect'].sum()/g['TotHandle'].sum()*100,4)
                  if g['TotHandle'].sum()>0 else np.nan,
        'Handles':int(g['Prime_Handle'].sum()),
    })).reset_index().dropna(subset=['FPY'])
    line_a['Gap'] = (line_a['FPY'] - line_a['FTY']).round(4)

    # Process FPY/FTY side-by-side
    st.markdown('<div class="sh">⚙️ Process — FPY vs FTY vs Fail%</div>',unsafe_allow_html=True)
    p1,p2 = st.columns(2)
    with p1:
        fig_p1 = go.Figure()
        fig_p1.add_trace(go.Bar(x=proc_a['Process'],y=proc_a['FPY'],name='FPY',
            marker_color=BLUE,text=proc_a['FPY'].round(2).astype(str)+'%',textposition='outside'))
        fig_p1.add_trace(go.Bar(x=proc_a['Process'],y=proc_a['FTY'],name='FTY',
            marker_color=GREEN,text=proc_a['FTY'].round(2).astype(str)+'%',textposition='outside'))
        fig_p1.add_hline(y=target_fpy,line_dash="dash",line_color=RED)
        fig_p1.update_layout(barmode='group',title='FPY & FTY by Process',
            yaxis=dict(range=[0,112]),height=340,margin=dict(l=15,r=10,t=45,b=40),
            legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig_p1,use_container_width=True)
    with p2:
        fig_p2 = go.Figure()
        fig_p2.add_trace(go.Bar(x=proc_a['Process'],y=proc_a['FailPct'],name='Fail%',
            marker_color=RED,text=proc_a['FailPct'].round(2).astype(str)+'%',textposition='outside'))
        fig_p2.add_trace(go.Bar(x=proc_a['Process'],y=proc_a['NTFPct'],name='NTF%',
            marker_color=ORANGE,text=proc_a['NTFPct'].round(2).astype(str)+'%',textposition='outside'))
        fig_p2.update_layout(barmode='group',title='Process: Fail% vs NTF%',
            height=340,margin=dict(l=15,r=10,t=45,b=40),
            legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig_p2,use_container_width=True)

    # NTF count
    st.markdown('<div class="sh">🔘 NTF Count by Process</div>',unsafe_allow_html=True)
    fig_ntf = px.bar(proc_a.sort_values('NTFCount',ascending=False),
        x='Process',y='NTFCount',color='NTFCount',
        color_continuous_scale='Oranges',text='NTFCount',title='NTF Count by Process')
    fig_ntf.update_traces(textposition='outside')
    fig_ntf.update_layout(height=280,margin=dict(l=15,r=15,t=40,b=40),showlegend=False)
    st.plotly_chart(fig_ntf,use_container_width=True)

    # Line FPY/FTY/Gap dual axis
    st.markdown('<div class="sh">🏭 Line — FPY vs FTY vs Gap</div>',unsafe_allow_html=True)
    fig_l = go.Figure()
    fig_l.add_trace(go.Bar(x=line_a['Line'],y=line_a['FPY'],name='FPY',
        marker_color=BLUE,text=line_a['FPY'].round(2).astype(str)+'%',textposition='outside'))
    fig_l.add_trace(go.Bar(x=line_a['Line'],y=line_a['FTY'],name='FTY',
        marker_color=GREEN,text=line_a['FTY'].round(2).astype(str)+'%',textposition='outside'))
    fig_l.add_trace(go.Scatter(x=line_a['Line'],y=line_a['Gap'],
        mode='lines+markers',name='Gap',line=dict(color=ORANGE,width=2.5),yaxis='y2'))
    fig_l.add_trace(go.Scatter(x=line_a['Line'],y=line_a['FailPct'],
        mode='lines+markers',name='Fail%',line=dict(color=RED,width=1.5,dash='dot'),yaxis='y2'))
    fig_l.update_layout(barmode='group',
        yaxis=dict(range=[0,112],title='Yield %'),
        yaxis2=dict(overlaying='y',side='right',title='Gap / Fail %',range=[0,15]),
        height=370,margin=dict(l=30,r=55,t=15,b=40),
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    fig_l.add_hline(y=target_fpy,line_dash="dash",line_color=RED)
    st.plotly_chart(fig_l,use_container_width=True)

    # Heatmap Line × Process
    st.markdown('<div class="sh">🌡️ Heatmap — Line × Process</div>',unsafe_allow_html=True)
    hm_met = st.selectbox("Metric",["FPY","FTY","FailPct","NTFPct"],key='hm_lp')
    lp = df.groupby(['Line','Process']).apply(lambda g: pd.Series({
        'FPY':    round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,2)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'FTY':    round(g['Prime_Pass'].sum()/g['Prime_Handle'].sum()*100,2)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'FailPct':round(g['Prime_Fail'].sum()/g['Prime_Handle'].sum()*100,2)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'NTFPct': round(g['PrimeCount'].sum()/g['Prime_Handle'].sum()*100,2)
                  if g['Prime_Handle'].sum()>0 else np.nan,
    })).reset_index()
    pv_lp = lp.pivot(index='Line',columns='Process',values=hm_met)
    scale = 'RdYlGn' if hm_met in ['FPY','FTY'] else 'RdYlGn_r'
    fig_hm = px.imshow(pv_lp,text_auto='.2f',color_continuous_scale=scale,
        aspect='auto',labels=dict(color=hm_met+' %'))
    fig_hm.update_layout(height=max(220,len(pv_lp)*46+90),margin=dict(l=10,r=10,t=15,b=15))
    st.plotly_chart(fig_hm,use_container_width=True)

    # Bubble: Process × Line × Time
    if 'TimePeriod' in df.columns:
        st.markdown('<div class="sh">🫧 Process × Line × Hour — Bubble</div>',unsafe_allow_html=True)
        bub = df.groupby(['Process','Line','TimePeriod']).apply(lambda g: pd.Series({
            'FPY':    round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,2)
                      if g['Prime_Handle'].sum()>0 else np.nan,
            'Handles':g['Prime_Handle'].sum(),
            'FailPct':round(g['Prime_Fail'].sum()/g['Prime_Handle'].sum()*100,2)
                      if g['Prime_Handle'].sum()>0 else np.nan,
        })).reset_index().dropna(subset=['FPY'])
        fig_bub = px.scatter(bub,x='TimePeriod',y='FPY',size='Handles',color='Process',
            symbol='Line',hover_data=['FailPct','Handles'],
            title='Process × Line × Hour (bubble=handles)',
            color_discrete_sequence=px.colors.qualitative.Bold)
        fig_bub.add_hline(y=target_fpy,line_dash="dash",line_color=RED)
        fig_bub.update_layout(height=380,margin=dict(l=30,r=20,t=50,b=40))
        st.plotly_chart(fig_bub,use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — TECHNOLOGY · FAMILY
# ════════════════════════════════════════════════════════════════
with T3:
    tech_a = df.groupby('Technology').apply(lambda g: pd.Series({
        'FPY':    round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,4)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'FTY':    round(g['Prime_Pass'].sum()/g['Prime_Handle'].sum()*100,4)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'FailPct':round(g['Prime_Fail'].sum()/g['Prime_Handle'].sum()*100,4)
                  if g['Prime_Handle'].sum()>0 else np.nan,
        'Handles':int(g['Prime_Handle'].sum()),
    })).reset_index().dropna(subset=['FPY'])

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sh">🔬 Failure Rate by Technology</div>',unsafe_allow_html=True)
        fig_tf = px.bar(tech_a.sort_values('FailPct',ascending=False),
            x='Technology',y='FailPct',color='FailPct',
            color_continuous_scale='Reds',text=tech_a.sort_values('FailPct',ascending=False)['FailPct'].round(2).astype(str)+'%',
            title='Fail% by Technology')
        fig_tf.update_traces(textposition='outside')
        fig_tf.update_layout(height=320,margin=dict(l=15,r=15,t=45,b=40),showlegend=False)
        st.plotly_chart(fig_tf,use_container_width=True)
    with c2:
        st.markdown('<div class="sh">🕸 Radar — FPY vs FTY by Technology</div>',unsafe_allow_html=True)
        cats = list(tech_a['Technology'])
        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(r=tech_a['FPY'],theta=cats,fill='toself',
            name='FPY',line_color=BLUE))
        fig_rad.add_trace(go.Scatterpolar(r=tech_a['FTY'],theta=cats,fill='toself',
            name='FTY',line_color=GREEN,opacity=.65))
        fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True,range=[80,100])),
            height=320,margin=dict(l=40,r=40,t=30,b=30),
            legend=dict(orientation="h",yanchor="bottom",y=-.1))
        st.plotly_chart(fig_rad,use_container_width=True)

    # Box plot
    st.markdown('<div class="sh">📦 Box Plot — Technology vs Row-Level FPY</div>',unsafe_allow_html=True)
    df_box = df.copy()
    df_box['row_FPY'] = np.where(df_box['Prime_Handle']>0,
        (df_box['Prime_Pass']+df_box['PrimeCount'])/df_box['Prime_Handle']*100, np.nan)
    fig_box = px.box(df_box.dropna(subset=['row_FPY','Technology']),
        x='Technology',y='row_FPY',color='Technology',points='all',
        color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_box.add_hline(y=target_fpy,line_dash="dash",line_color=RED,
        annotation_text=f"Target {target_fpy}%")
    fig_box.update_layout(height=360,showlegend=False,margin=dict(l=15,r=15,t=15,b=40))
    st.plotly_chart(fig_box,use_container_width=True)

    # Family
    st.markdown('<div class="sh">👨‍👩‍👧 Family — FPY & FTY</div>',unsafe_allow_html=True)
    fam_a = df.groupby('Family').apply(lambda g: pd.Series({
        'FPY': round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,4)
               if g['Prime_Handle'].sum()>0 else np.nan,
        'FTY': round(g['Prime_Pass'].sum()/g['Prime_Handle'].sum()*100,4)
               if g['Prime_Handle'].sum()>0 else np.nan,
        'Handles':int(g['Prime_Handle'].sum()),
    })).reset_index().dropna(subset=['FPY']).sort_values('Handles',ascending=False)

    fig_fam = go.Figure()
    fig_fam.add_trace(go.Bar(y=fam_a['Family'],x=fam_a['FPY'],name='FPY',orientation='h',
        marker_color=BLUE,text=fam_a['FPY'].round(2).astype(str)+'%',textposition='outside'))
    fig_fam.add_trace(go.Bar(y=fam_a['Family'],x=fam_a['FTY'],name='FTY',orientation='h',
        marker_color=GREEN,opacity=.8,text=fam_a['FTY'].round(2).astype(str)+'%',textposition='outside'))
    fig_fam.add_vline(x=target_fpy,line_dash="dash",line_color=RED)
    fig_fam.update_layout(barmode='group',xaxis=dict(range=[0,112]),
        height=max(260,len(fam_a)*48+80),margin=dict(l=10,r=45,t=15,b=30),
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    st.plotly_chart(fig_fam,use_container_width=True)

    # Sunburst
    st.markdown('<div class="sh">🌞 Sunburst — Technology → Process → Line</div>',unsafe_allow_html=True)
    sun = df.groupby(['Technology','Process','Line']).apply(lambda g: pd.Series({
        'Handles': int(g['Prime_Handle'].sum()),
        'FPY': round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,2)
               if g['Prime_Handle'].sum()>0 else 0,
    })).reset_index()
    sun = sun[sun['Handles']>0]
    if not sun.empty:
        fig_sun = px.sunburst(sun,path=['Technology','Process','Line'],
            values='Handles',color='FPY',color_continuous_scale='RdYlGn',range_color=[80,100],
            title='Handle Volume & FPY')
        fig_sun.update_layout(height=460,margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig_sun,use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — ALT FTY/FPY INSIGHT (Image-3 formula)
# ════════════════════════════════════════════════════════════════
with T4:
    st.markdown('<div class="sh">📐 Alt-FTY Insight: FTY = Pass / Max(PH across all processes)</div>',unsafe_allow_html=True)
    st.info("""**Formula (from Image-3):**  
• **FTY_alt** = Prime_Pass / max(Prime_Handle across all processes) × 100  — uses the largest handle count as common denominator  
• **FPY_alt** = (Prime_Pass + NTF) / Prime_Handle × 100  — same as standard FPY  
• **Overall FTY_alt** = ∏(per-process FTY_alt fractions) × 100""")

    # By Process
    st.markdown('<div class="sh-g">⚙️ Alt-FTY / FPY by Process</div>',unsafe_allow_html=True)
    alt_proc = alt_yield_by_process(df)
    if not alt_proc.empty:
        fig_alt = go.Figure()
        fig_alt.add_trace(go.Bar(x=alt_proc['Process'],y=alt_proc['FPY_alt'],name='FPY (std)',
            marker_color=BLUE,text=alt_proc['FPY_alt'].round(2).astype(str)+'%',textposition='outside'))
        fig_alt.add_trace(go.Bar(x=alt_proc['Process'],y=alt_proc['FTY_std'],name='FTY (std)',
            marker_color=GREEN,text=alt_proc['FTY_std'].round(2).astype(str)+'%',textposition='outside'))
        fig_alt.add_trace(go.Scatter(x=alt_proc['Process'],y=alt_proc['FTY_alt'],
            mode='lines+markers',name='FTY_alt (shared denom)',
            line=dict(color=PURPLE,width=2.5,dash='dot'),marker=dict(size=9)))
        fig_alt.add_hline(y=target_fpy,line_dash="dash",line_color=RED)
        fig_alt.update_layout(barmode='group',yaxis=dict(range=[0,115],title='Yield %'),
            height=360,margin=dict(l=20,r=20,t=20,b=45),
            legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig_alt,use_container_width=True)

        # Table
        alt_disp = alt_proc[['Process','Prime_Handle','Prime_Pass','NTF','FTY_std','FPY_alt','FTY_alt']].copy()
        alt_disp.columns = ['Process','Prime_Handle','Prime_Pass','NTF','FTY_std%','FPY_alt%','FTY_alt%']
        max_ph_val = df['Prime_Handle'].sum()
        st.caption(f"Common denominator (max Prime_Handle across all processes): **{int(max_ph_val):,}**")
        st.dataframe(alt_disp.round(4),use_container_width=True,hide_index=True)

    # By Line
    st.markdown('<div class="sh-g">🏭 Alt-FTY / FPY by Line</div>',unsafe_allow_html=True)
    alt_line = alt_yield_by_line(df)
    if not alt_line.empty:
        fig_al = go.Figure()
        fig_al.add_trace(go.Bar(x=alt_line['Line'],y=alt_line['FPY_alt'],name='FPY (std)',
            marker_color=BLUE,text=alt_line['FPY_alt'].round(2).astype(str)+'%',textposition='outside'))
        fig_al.add_trace(go.Bar(x=alt_line['Line'],y=alt_line['FTY_std'],name='FTY (std)',
            marker_color=GREEN,text=alt_line['FTY_std'].round(2).astype(str)+'%',textposition='outside'))
        fig_al.add_trace(go.Scatter(x=alt_line['Line'],y=alt_line['FTY_alt'],
            mode='lines+markers',name='FTY_alt (shared denom)',
            line=dict(color=PURPLE,width=2.5,dash='dot'),marker=dict(size=9)))
        fig_al.add_hline(y=target_fpy,line_dash="dash",line_color=RED)
        fig_al.update_layout(barmode='group',yaxis=dict(range=[0,115]),
            height=360,margin=dict(l=20,r=20,t=20,b=45),
            legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig_al,use_container_width=True)

    # TestCode FPY insight
    if not hf.empty and 'TestCode' in hf.columns:
        st.markdown('<div class="sh-g">🔬 TestCode — FPY Insight (PFail rate per Handle)</div>',unsafe_allow_html=True)
        tc_ph = df['Prime_Handle'].sum()
        tc_ins = hf.groupby(['TestCode','Process']).agg(
            PFail=('PFail','sum'),TFail=('TFail','sum'),PHandle=('PHandle','sum')
        ).reset_index()
        tc_ins['FailRate'] = np.where(tc_ph>0, tc_ins['PFail']/tc_ph*100, np.nan)
        tc_ins = tc_ins.sort_values('PFail',ascending=False).head(15)
        fig_tci = px.bar(tc_ins,x='TestCode',y='PFail',color='Process',
            text=tc_ins['PFail'].astype(str),
            title='TestCode PFail (relation to process)',
            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_tci.update_traces(textposition='outside')
        fig_tci.update_layout(height=340,margin=dict(l=15,r=15,t=45,b=55))
        st.plotly_chart(fig_tci,use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 5 — DEFECT · NTF · DPHU
# ════════════════════════════════════════════════════════════════
with T5:
    # NTF vs Fail scatter
    st.markdown('<div class="sh">🔘 NTF vs Fail Count — Process View</div>',unsafe_allow_html=True)
    nf_sc = df.groupby(['Process','Line']).apply(lambda g: pd.Series({
        'NTFCount': int(g['PrimeCount'].sum()),
        'FailCount':int(g['Prime_Fail'].sum()),
        'Handles':  g['Prime_Handle'].sum(),
        'DPHU':     round(g['TotalDefect'].sum()/g['TotHandle'].sum()*100,2)
                    if g['TotHandle'].sum()>0 else 0,
        'FPY':      round((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100,2)
                    if g['Prime_Handle'].sum()>0 else np.nan,
    })).reset_index().dropna(subset=['FPY'])
    fig_nf = px.scatter(nf_sc,x='FailCount',y='NTFCount',size='Handles',color='Process',
        text='Line',title='NTF vs Fail (bubble=handles)',
        labels={'FailCount':'Prime Fails','NTFCount':'NTF Count'},
        color_discrete_sequence=px.colors.qualitative.Bold)
    fig_nf.update_traces(textposition='top center')
    fig_nf.update_layout(height=360,margin=dict(l=30,r=20,t=50,b=40))
    st.plotly_chart(fig_nf,use_container_width=True)

    # DPHU vs FPY scatter
    # Replace the old fig_ds block with this:

    st.markdown('<div class="sh">📉 DPHU vs FPY — Line × Process</div>', unsafe_allow_html=True)

    dphu_sc = df.groupby(['Line','Process']).apply(lambda g: pd.Series({
        'FPY': round((g['Prime_Pass'].sum() + g['PrimeCount'].sum()) / 
                 g['Prime_Handle'].sum() * 100, 2) if g['Prime_Handle'].sum() > 0 else np.nan,
        'DPHU': round(g['TotalDefect'].sum() / g['TotHandle'].sum() * 100, 2) 
            if g['TotHandle'].sum() > 0 else 0,
        'Handles': g['Prime_Handle'].sum(),
    })).reset_index().dropna(subset=['FPY'])

    fig_ds = px.scatter(dphu_sc, x='DPHU', y='FPY', size='Handles', color='Process',
                    text='Line', title='DPHU vs FPY (Bubble = Handles)',
                    color_discrete_sequence=px.colors.qualitative.Safe)

# Manual trend line (more reliable)
    if len(dphu_sc) > 5:
      import numpy as np
      z = np.polyfit(dphu_sc['DPHU'], dphu_sc['FPY'], 1)
      p = np.poly1d(z)
      x_trend = np.linspace(dphu_sc['DPHU'].min(), dphu_sc['DPHU'].max(), 100)
      fig_ds.add_trace(go.Scatter(
        x=x_trend, y=p(x_trend),
        mode='lines',
        name='Trend (OLS)',
        line=dict(color='red', dash='dash', width=2.5)
    ))

    fig_ds.add_hline(y=target_fpy, line_dash="dash", line_color="red", 
                 annotation_text=f"Target {target_fpy}%")
    fig_ds.update_layout(height=380, margin=dict(l=30,r=20,t=50,b=40))

    st.plotly_chart(fig_ds, use_container_width=True)

    # Defect mapping from hidden
    if not hf.empty and all(c in hf.columns for c in ['TestCode','Process','TFail','TDef']):
        st.markdown('<div class="sh">🗺 TestCode → Process Defect Treemap</div>',unsafe_allow_html=True)
        def_map = hf.groupby(['TestCode','Process']).agg(
            TFail=('TFail','sum'),TDef=('TDef','sum')).reset_index()
        def_map = def_map[def_map['TFail']>0].sort_values('TFail',ascending=False).head(20)
        if not def_map.empty:
            fig_dm = px.treemap(def_map,path=['Process','TestCode'],values='TFail',
                color='TDef',color_continuous_scale='OrRd',
                title='TestCode Failures by Process (size=TFail, color=TDef)')
            fig_dm.update_layout(height=420,margin=dict(l=10,r=10,t=45,b=10))
            st.plotly_chart(fig_dm,use_container_width=True)

            d1,d2 = st.columns(2)
            with d1:
                fig_tc = go.Figure()
                fig_tc.add_trace(go.Bar(x=def_map['TestCode'],y=def_map['TFail'],
                    name='TFail',marker_color=RED))
                fig_tc.add_trace(go.Bar(x=def_map['TestCode'],y=def_map['TDef'],
                    name='TDef',marker_color=ORANGE,opacity=.8))
                fig_tc.update_layout(barmode='group',title='Top TestCodes: TFail vs TDef',
                    height=300,xaxis_tickangle=-35,margin=dict(l=15,r=10,t=45,b=65),
                    legend=dict(orientation="h",yanchor="bottom",y=1.02))
                st.plotly_chart(fig_tc,use_container_width=True)
            with d2:
                pd_f = hf.groupby('Process').agg(TFail=('TFail','sum'),TDef=('TDef','sum')).reset_index()
                fig_pd = go.Figure()
                fig_pd.add_trace(go.Bar(x=pd_f['Process'],y=pd_f['TFail'],name='TFail',marker_color=RED))
                fig_pd.add_trace(go.Bar(x=pd_f['Process'],y=pd_f['TDef'],name='TDef',marker_color=ORANGE,opacity=.8))
                fig_pd.update_layout(barmode='group',title='Process: TFail vs TDef',
                    height=300,margin=dict(l=15,r=10,t=45,b=40),
                    legend=dict(orientation="h",yanchor="bottom",y=1.02))
                st.plotly_chart(fig_pd,use_container_width=True)

    # Process vs Fail vs Defect vs NTF combined
    st.markdown('<div class="sh">📊 Process — Fail vs Defect vs NTF (Combined)</div>',unsafe_allow_html=True)
    pfd = df.groupby('Process').apply(lambda g: pd.Series({
        'Failures':int(g['Prime_Fail'].sum()),
        'Defects': int(g['TotalDefect'].sum()),
        'NTF':     int(g['PrimeCount'].sum()),
    })).reset_index()
    fig_pfd = go.Figure()
    fig_pfd.add_trace(go.Bar(name='Prime Fails',x=pfd['Process'],y=pfd['Failures'],marker_color=RED))
    fig_pfd.add_trace(go.Bar(name='Total Defects',x=pfd['Process'],y=pfd['Defects'],marker_color=ORANGE))
    fig_pfd.add_trace(go.Bar(name='NTF Count',x=pfd['Process'],y=pfd['NTF'],marker_color=YELLOW))
    fig_pfd.update_layout(barmode='group',height=320,
        margin=dict(l=15,r=10,t=15,b=40),
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    st.plotly_chart(fig_pfd,use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 6 — PARETO · RTY
# ════════════════════════════════════════════════════════════════
with T6:
    # Pareto: Process vs Yield Loss
    st.markdown('<div class="sh">📉 Pareto — Process vs Yield Loss</div>',unsafe_allow_html=True)
    par_p = proc_a[['Process','FPY']].copy()
    par_p['YieldLoss'] = (100 - par_p['FPY']).round(4)
    par_p = par_p[par_p['YieldLoss']>0]
    st.plotly_chart(pareto_fig(par_p,'Process','YieldLoss','Pareto: Process vs Yield Loss (80/20)'),
        use_container_width=True)

    # Pareto: Line vs Fail
    st.markdown('<div class="sh">📉 Pareto — Line vs Prime Fail Count</div>',unsafe_allow_html=True)
    par_l = df.groupby('Line').agg(FailCount=('Prime_Fail','sum')).reset_index()
    st.plotly_chart(pareto_fig(par_l,'Line','FailCount','Pareto: Line vs Prime Fail Count'),
        use_container_width=True)

    # Pareto: TestCode
    if not hf.empty and 'TestCode' in hf.columns:
        st.markdown('<div class="sh">📉 Pareto — TestCode vs TFail</div>',unsafe_allow_html=True)
        par_tc = hf.groupby('TestCode').agg(TFail=('TFail','sum')).reset_index()
        st.plotly_chart(pareto_fig(par_tc,'TestCode','TFail','Pareto: TestCode vs TFail'),
            use_container_width=True)

    # RTY waterfall
    st.markdown('<div class="sh">🔄 RTY — Rolled Throughput Yield Across Processes</div>',unsafe_allow_html=True)
    if not rty_df.empty:
        fig_rty = make_subplots(specs=[[{"secondary_y":True}]])
        fig_rty.add_trace(go.Bar(x=rty_df['Process'],y=rty_df['FPY_pct'],name='FPY %',
            marker_color=BLUE,text=rty_df['FPY_pct'].round(2).astype(str)+'%',
            textposition='outside'),secondary_y=False)
        fig_rty.add_trace(go.Scatter(x=rty_df['Process'],y=rty_df['RTY_running'],
            mode='lines+markers',name='Running RTY',
            line=dict(color=RED,width=3),marker=dict(size=9)),secondary_y=True)
        fig_rty.update_yaxes(title_text="FPY %",range=[85,105],secondary_y=False)
        fig_rty.update_yaxes(title_text="Cumulative RTY %",range=[60,105],secondary_y=True)
        fig_rty.update_layout(title=f"RTY Degradation (Final RTY = {OV_RTY:.2f}%)",
            height=370,xaxis_tickangle=-35,margin=dict(l=35,r=55,t=55,b=75),
            legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig_rty,use_container_width=True)

        r1,r2,r3 = st.columns(3)
        r1.metric("Final RTY", f"{OV_RTY:.2f}%")
        r2.metric("Processes in Chain", len(rty_df))
        r3.metric("Avg FPY per Process", f"{rty_df['FPY_pct'].mean():.2f}%")
    else:
        st.info("RTY requires process-level data. Check filters.")

    # Recovery
    st.markdown('<div class="sh">💊 Recovery = (FTY−FPY)/(1−FPY) per Line</div>',unsafe_allow_html=True)
    rec_d = line_a[['Line','FPY','FTY']].copy()
    # Clamp: recovery meaningful only when FPY < 99.5
    rec_d['Recovery'] = np.where(rec_d['FPY']<99.5,
        (rec_d['FTY']-rec_d['FPY'])/(100-rec_d['FPY'])*100, 0.0).round(2)
    rec_d['Recovery'] = rec_d['Recovery'].clip(-100,100)
    rec_d = rec_d.sort_values('Recovery',ascending=False)
    fig_rec = px.bar(rec_d,x='Line',y='Recovery',color='Recovery',
        color_continuous_scale=['#ef4444','#fbbf24','#86efac','#22c55e'],
        text=rec_d['Recovery'].astype(str)+'%',
        title='Recovery % per Line')
    fig_rec.update_traces(textposition='outside')
    fig_rec.add_hline(y=0,line_dash="dash",line_color="gray",line_width=2)
    fig_rec.update_layout(height=360,yaxis=dict(range=[-110,110]),
        margin=dict(l=15,r=15,t=50,b=55),showlegend=False)
    st.plotly_chart(fig_rec,use_container_width=True)
    st.info("**Positive** → rework is recovering units. **Negative** → failures worsen after rework. **Near 0** → rework has little effect.")

# ════════════════════════════════════════════════════════════════
# TAB 7 — RAW DATA
# ════════════════════════════════════════════════════════════════
with T7:
    tbl = st.radio("Table",["rld1 (filtered)","rld1_hidden (filtered)"],horizontal=True)
    if tbl.startswith("rld1 ("):
        disp = df.copy()
        disp['row_FPY'] = np.where(disp['Prime_Handle']>0,
            (disp['Prime_Pass']+disp['PrimeCount'])/disp['Prime_Handle']*100, np.nan).round(4)
        disp['row_FTY'] = np.where(disp['Prime_Handle']>0,
            disp['Prime_Pass']/disp['Prime_Handle']*100, np.nan).round(4)
        disp['row_DPHU']= np.where(disp['TotHandle']>0,
            disp['TotalDefect']/disp['TotHandle']*100, np.nan).round(4)
        show = [c for c in ['Date','SlotLabel','TimePeriod','Line','Process','Technology','Family',
                             'Prime_Pass','Prime_Fail','Prime_Handle','PrimeCount','row_FPY','row_FTY',
                             'TotalDefect','TotalNTFCount','row_DPHU'] if c in disp.columns]
        st.dataframe(disp[show].sort_values(['Date','SlotLabel','Line'],na_position='last'),
                     use_container_width=True,height=500)
    else:
        st.dataframe(hf.sort_values('TFail',ascending=False,na_position='last'),
                     use_container_width=True,height=500)

st.markdown("---")
st.caption(f"🔄 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | FPY=(Pass+NTF)/Handle×100 | FTY=Pass/Handle×100 | DPHU=Defects/TotHandle×100 | RTY=∏(per-process FPY fracs)×100 | Recovery=(FTY−FPY)/(100−FPY)×100")
