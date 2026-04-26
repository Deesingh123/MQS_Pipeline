"""
MQS Yield Intelligence v5.1 — Industrial Excellence with ₹ & Corrected Analytics
Shift fix · Centering advice · COPQ correction · OEE loss tree · Root cause insights
"""
import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import psycopg2, os, math, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

st.set_page_config(page_title="MQS Industrial Excellence", layout="wide",
                   page_icon="🏭", initial_sidebar_state="expanded")
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300_000, key="yr")
except ImportError:
    pass

# ======================== COLOR THEME ==========================
B = "#1e3a8a"; C = "#0891b2"; G = "#10b981"; R = "#ef4444"; O = "#f59e0b"
P = "#8b5cf6"; T = "#14b8a6"; GR = "#64748b"; D = "#0f172a"; BR = "#e2e8f0"

st.markdown(f"""<style>
.stApp{{background:linear-gradient(135deg,#f8fafc,#f1f5f9)}}
.block-container{{padding-top:.4rem;padding-bottom:.4rem}}
@media(max-width:768px){{.block-container{{padding:.25rem}}
  div[data-testid="stMetricValue"]{{font-size:1.1rem!important}}}}
div[data-testid="stMetric"]{{background:rgba(255,255,255,.93);border:1px solid {BR};
  border-radius:14px;padding:10px 14px;box-shadow:0 4px 12px rgba(0,0,0,.05)}}
div[data-testid="stMetric"] label{{color:{GR}!important;font-size:.68rem;font-weight:600}}
div[data-testid="stMetricValue"]{{font-size:1.65rem!important;font-weight:700;color:{D}}}
.sh{{background:linear-gradient(90deg,{B}14,transparent);border-left:4px solid {B};
  padding:7px 14px;border-radius:0 10px 10px 0;margin:.5rem 0;font-weight:700;color:{B}}}
.shr{{background:linear-gradient(90deg,{R}14,transparent);border-left:4px solid {R};
  padding:7px 14px;border-radius:0 10px 10px 0;margin:.5rem 0;font-weight:700;color:{R}}}
.shg{{background:linear-gradient(90deg,{G}14,transparent);border-left:4px solid {G};
  padding:7px 14px;border-radius:0 10px 10px 0;margin:.5rem 0;font-weight:700;color:#15803d}}
</style>""", unsafe_allow_html=True)

# ======================== DATABASE =============================
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

def enrich(df):
    d = df.copy()
    num_cols = ['Prime_Pass','Prime_Fail','Prime_Handle','TotPass','TotFail','TotHandle',
                'PrimeCount','TotalNTFCount','PrimeDefectCount','TotalDefect','TotalDPHU','TimePeriod']
    for c in num_cols:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0)
    if 'Date' in d.columns: d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
    
    d['FTY'] = np.where(d['Prime_Handle'] > 0, d['Prime_Pass'] / d['Prime_Handle'] * 100, np.nan)
    d['FPY'] = np.where(d['Prime_Handle'] > 0, (d['Prime_Pass'] + d['PrimeCount']) / d['Prime_Handle'] * 100, np.nan)
    d['HFI'] = np.where((100 - d['FTY']) > 0, (d['FPY'] - d['FTY']) / (100 - d['FTY']) * 100, 0)
    d['HFI'] = d['HFI'].clip(lower=0, upper=100)
    d['FailPct'] = np.where(d['Prime_Handle'] > 0, d['Prime_Fail'] / d['Prime_Handle'] * 100, np.nan)
    d['NTFPct'] = np.where(d['Prime_Handle'] > 0, d['PrimeCount'] / d['Prime_Handle'] * 100, np.nan)
    d['DPMO'] = np.where(d['Prime_Handle'] > 0, (d['Prime_Fail'] / d['Prime_Handle']) * 1_000_000, np.nan)
    
    # Correct shift classification: Night (22-6), Morning (6-14), Afternoon (14-22)
    if 'TimePeriod' in d.columns:
        def shift_from_hour(h):
            if pd.isna(h): return 'Unknown'
            if h >= 22 or h < 6: return 'Night'
            elif h < 14: return 'Morning'
            else: return 'Afternoon'
        d['Shift'] = d['TimePeriod'].apply(shift_from_hour)
    return d

# ======================== STATISTICAL FUNCTIONS =================
def cpk(s, usl=100, lsl=0):
    if len(s) < 2 or s.std() == 0: return 0
    m, sd = s.mean(), s.std()
    return min((usl-m)/(3*sd), (m-lsl)/(3*sd))

def ppk(s, usl=100, lsl=0):
    if len(s) < 2 or s.std() == 0: return 0
    m, sd = s.mean(), s.std()
    return min((usl-m)/(3*sd), (m-lsl)/(3*sd))

def dpmo_to_sigma(dpmo):
    from scipy.special import ndtri
    if dpmo <= 0: return 6.0
    return ndtri(1 - dpmo/1_000_000) + 1.5

def oee(availability, performance, quality):
    return availability * performance * quality

# ======================== KPI AGGREGATION ======================
def kpis(df):
    PP = df['Prime_Pass'].sum(); PH = df['Prime_Handle'].sum()
    NTF = df['PrimeCount'].sum(); PF = df['Prime_Fail'].sum()
    TH = df['TotHandle'].sum(); TD = df['TotalDefect'].sum()
    fpy = (PP+NTF)/PH*100 if PH>0 else 0
    fty = PP/PH*100 if PH>0 else 0
    hfi = df['HFI'].mean() if 'HFI' in df.columns else 0
    dpmo = (PF / PH) * 1_000_000 if PH>0 else 0
    sigma = dpmo_to_sigma(dpmo)
    return dict(FPY=round(fpy,4), FTY=round(fty,4), DPHU=round(TD/TH*100,4) if TH>0 else 0,
                Fail=round(PF/PH*100,4) if PH>0 else 0, NTFp=round(NTF/PH*100,4) if PH>0 else 0,
                PH=int(PH), PF=int(PF), NTF=int(NTF), TD=int(TD),
                Gap=round(fpy - fty,4), HFI=round(hfi,4),
                DPMO=int(dpmo), Sigma=round(sigma,2))

def rty_calc(df):
    rows = []
    for p, g in df.groupby('Process'):
        ph = g['Prime_Handle'].sum()
        if ph <= 0: continue
        fpy_frac = (g['Prime_Pass'].sum() + g['PrimeCount'].sum()) / ph
        rows.append({'Process':p, 'FPY_pct':fpy_frac*100, 'FPY_frac':fpy_frac})
    if not rows: return pd.DataFrame(), 0.0
    rdf = pd.DataFrame(rows).sort_values('FPY_pct', ascending=False)
    rty = round(float(np.prod(rdf['FPY_frac'])) * 100, 4)
    rdf['RTY_run'] = (rdf['FPY_frac'].cumprod() * 100).round(4)
    return rdf, rty

# ======================== LOAD & FILTER ========================
df_raw, hdf_raw = load_data()
if df_raw.empty:
    st.warning("No data – run scraper first.")
    st.stop()

df_all = enrich(df_raw)
hdf_all = hdf_raw.copy()
for c in ['PFail','TFail','PDef','TDef','PHandle','THandle']:
    if c in hdf_all.columns:
        hdf_all[c] = pd.to_numeric(hdf_all[c], errors='coerce').fillna(0)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    dates = sorted(df_all['Date'].dropna().dt.date.unique(), reverse=True)
    sd = st.multiselect("📅 Date(s)", dates, default=list(dates[:7]) if dates else [])
    all_sl = sorted(df_all['SlotLabel'].dropna().unique())
    ss = st.multiselect("⏱ Slot(s)", all_sl, default=all_sl)
    all_ln = sorted(df_all['Line'].dropna().unique())
    sl = st.multiselect("🏭 Line(s)", all_ln, default=all_ln)
    all_pr = sorted(df_all['Process'].dropna().unique())
    sp = st.multiselect("⚙️ Process", all_pr, default=all_pr)
    all_te = sorted(df_all['Technology'].dropna().unique())
    st_ = st.multiselect("🔬 Technology", all_te, default=all_te)
    all_fa = sorted(df_all['Family'].dropna().unique())
    sf = st.multiselect("👨‍👩‍👧 Family", all_fa, default=all_fa)
    st.markdown("---")
    tfpy = st.slider("🎯 FPY Target (%)", 80, 100, 95)
    tfty = st.slider("🎯 FTY Target (%)", 80, 100, 97)
    target_cpk = st.slider("Cpk Target", 0.5, 2.0, 1.33, 0.01)
    st.markdown("### 🏭 OEE / TEEP Inputs")
    planned_prod_time = st.number_input("Planned production time (hours/day)", 8.0, 24.0, 22.0)
    total_time = st.number_input("Total calendar time (hours/day)", 24.0, 24.0, 24.0)
    ideal_cycle_time = st.number_input("Ideal cycle time (minutes/unit)", 0.5, 10.0, 2.0)
    customer_demand = st.number_input("Daily customer demand (units)", 100, 10000, 1000)
    st.markdown("### 💰 COPQ Parameters (₹)")
    labor_rate = st.number_input("Rework labor cost (₹/hour)", 50.0, 1000.0, 500.0)
    rework_time = st.number_input("Avg rework time (minutes)", 1.0, 60.0, 10.0)
    scrap_cost_per_unit = st.number_input("Scrap cost per unit (₹)", 0.0, 1000.0, 500.0)
    external_failure_rate = st.number_input("External failure rate (% of shipped)", 0.0, 10.0, 1.0)
    external_failure_cost_per_unit = st.number_input("External failure cost per unit (₹)", 0, 50000, 5000)
    st.markdown("---")
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear(); st.rerun()
    st.caption(f"rld1: {len(df_all):,} rows | hidden: {len(hdf_all):,}")

def filt(df):
    m = pd.Series([True]*len(df))
    if sd and 'Date' in df.columns: m &= df['Date'].dt.date.isin(sd)
    if ss and 'SlotLabel' in df.columns: m &= df['SlotLabel'].isin(ss)
    if sl and 'Line' in df.columns: m &= df['Line'].isin(sl)
    if sp and 'Process' in df.columns: m &= df['Process'].isin(sp)
    if st_ and 'Technology' in df.columns: m &= df['Technology'].isin(st_)
    if sf and 'Family' in df.columns: m &= df['Family'].isin(sf)
    return df[m].copy()

df = filt(df_all)
hf = hdf_all[hdf_all['Line'].isin(sl)].copy() if sl else hdf_all.copy()
if ss and 'SlotLabel' in hf.columns: hf = hf[hf['SlotLabel'].isin(ss)]
if df.empty: st.warning("No data for selected filters."); st.stop()

# --------------------- KPIs & Derived --------------------------
K = kpis(df)
rty_df, OV_RTY = rty_calc(df)
rolled_fty = math.prod([g['Prime_Pass'].sum()/g['Prime_Handle'].sum() 
                        for _,g in df.groupby('Process') if g['Prime_Handle'].sum()>0]) * 100 if len(df.groupby('Process'))>0 else 0
CPK = cpk(df['FTY'].dropna())
PPK = ppk(df['FTY'].dropna())
VOL = df['FPY'].dropna().std()/df['FPY'].dropna().mean()*100 if df['FPY'].dropna().mean()>0 else 0

# OEE / TEEP
if 'TimePeriod' in df.columns and 'Prime_Handle' in df.columns:
    total_units = df['Prime_Handle'].sum()
    active_hours = df['SlotLabel'].nunique()
    availability = min(1.0, active_hours / planned_prod_time) if planned_prod_time>0 else 1.0
    performance = min(1.0, (total_units * ideal_cycle_time/60) / active_hours) if active_hours>0 else 1.0
    quality = K['FTY'] / 100
    OEE = oee(availability, performance, quality)
    TEEP = OEE * (planned_prod_time / total_time) if total_time>0 else 0

# Takt Time
takt_time = (planned_prod_time * 60) / customer_demand if customer_demand>0 else 0

# COPQ (corrected)
hidden_units = K['PH'] * (K['HFI']/100)
rework_cost = hidden_units * (rework_time/60) * labor_rate
scrap_cost = K['PF'] * scrap_cost_per_unit
shipped_units = K['PH'] - K['PF']  # approximate
external_cost = shipped_units * (external_failure_rate/100) * external_failure_cost_per_unit
coq = rework_cost + scrap_cost + external_cost

# Centering advice (fixed: shift to target FTY)
def centering_advice(mean, target, sd, cpk_target=1.33):
    if sd <= 0: return 0.0, "Insufficient data"
    shift_needed = target - mean
    new_cpk = min((100-target)/(3*sd), (target-0)/(3*sd))
    if new_cpk >= cpk_target:
        msg = f"Shift mean by {shift_needed:.2f}% to reach {target}% (Cpk would be {new_cpk:.2f})"
    else:
        msg = f"Variation too high (σ={sd:.2f}). Need to reduce variation to achieve Cpk {cpk_target}"
    return shift_needed, msg

# Anomaly detection
if len(df) > 20:
    feat_cols = [c for c in ['FTY','FPY','FailPct','NTFPct'] if c in df.columns]
    if feat_cols:
        X = StandardScaler().fit_transform(df[feat_cols].fillna(0))
        df['Anom'] = (IsolationForest(contamination=0.08, random_state=42).fit_predict(X) == -1)
        NA = df['Anom'].sum()
    else:
        df['Anom'] = False; NA = 0
else:
    df['Anom'] = False; NA = 0

# ----------------------------------------------------------------
# HEADER
st.markdown(f"""<div style="text-align:center;padding:.6rem 1rem;
  background:linear-gradient(135deg,{B}10,{T}07); border-bottom:2px solid {B}25;
  margin-bottom:.7rem; border-radius:0 0 18px 18px">
  <h1 style="font-size:clamp(1.3rem,3.5vw,2rem);font-weight:900;
    background:linear-gradient(90deg,{B},{T});-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;margin:0">
    🏭 MQS INDUSTRIAL EXCELLENCE (₹)
  </h1>
  <div style="color:{GR};font-size:.78rem">
    Cpk·Ppk·DPMO·OEE·TEEP·Takt·FMEA·COPQ·CAPA·Shift Analytics
  </div></div>""", unsafe_allow_html=True)

# KPI Row 1
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🎯 FPY", f"{K['FPY']:.2f}%", delta=f"{K['FPY']-tfpy:+.1f}% vs target")
c2.metric("🔁 FTY", f"{K['FTY']:.2f}%", delta=f"{K['Gap']:+.2f}% gap")
c3.metric("🔄 RTY", f"{OV_RTY:.2f}%")
c4.metric("⚡ Cpk", f"{CPK:.2f}", delta="✓" if CPK>=target_cpk else "⚠️ low")
c5.metric("📊 Ppk", f"{PPK:.2f}", delta=f"diff={PPK-CPK:+.2f}")
c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("🏭 Hidden Factory", f"{K['HFI']:.1f}%")
c7.metric("💰 COPQ", f"₹{coq:,.0f}", delta=f"per {K['PH']:,} units")
c8.metric("📉 DPMO", f"{K['DPMO']:,}", delta=f"σ={K['Sigma']:.1f}")
c9.metric("🛡 DPHU", f"{K['DPHU']:.2f}")
c10.metric("⚠️ Anomalies", f"{NA}")

# OEE / TEEP / Takt
c11, c12, c13 = st.columns(3)
c11.metric("🏭 OEE", f"{OEE*100:.1f}%", help="Availability × Performance × Quality")
c12.metric("⏱️ TEEP", f"{TEEP*100:.1f}%", help="OEE × Utilization")
c13.metric("⏲️ Takt Time", f"{takt_time:.1f} min/unit", help=f"Demand {customer_demand} units/day")

# Centering alert
if CPK < target_cpk:
    shift_needed, msg = centering_advice(df['FTY'].dropna().mean(), tfty, df['FTY'].dropna().std(), target_cpk)
    st.warning(f"📌 **Centering Advice**: {msg} | Current Cpk={CPK:.2f} → Target {target_cpk:.2f}")

# ====================== TABS ===================================
tabs = st.tabs([
    "📈 Trends & Shift", "⚙️ Process·Line", "🔬 Tech·Family",
    "🎲 Monte Carlo", "🧮 Bayesian", "🔗 Markov",
    "📊 SPC·Cpk·Ppk", "🤖 Anomaly Clusters", "📉 Pareto·RTY",
    "🏭 OEE·TEEP", "📋 FMEA·CAPA", "💰 COPQ·DPMO", "🗂 Data"
])

# --------------------- TAB 0: Trends & Shift ---------------------
with tabs[0]:
    st.markdown('<div class="sh">📈 Slot Trend with Dynamic Limits</div>', unsafe_allow_html=True)
    slot_df = df.groupby('SlotLabel').apply(lambda g: pd.Series({
        'FPY': (g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100 if g['Prime_Handle'].sum()>0 else np.nan,
        'FTY': g['Prime_Pass'].sum()/g['Prime_Handle'].sum()*100 if g['Prime_Handle'].sum()>0 else np.nan,
    })).reset_index().dropna()
    if len(slot_df)>5:
        window = min(5, len(slot_df)//2)
        slot_df['MA'] = slot_df['FTY'].rolling(window, min_periods=1).mean()
        slot_df['sigma'] = slot_df['FTY'].rolling(window, min_periods=1).std()
        slot_df['UCL'] = slot_df['MA'] + 3*slot_df['sigma']
        slot_df['LCL'] = slot_df['MA'] - 3*slot_df['sigma']
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=slot_df['SlotLabel'], y=slot_df['FPY'], mode='lines+markers', name='FPY', line=dict(color=B)))
        fig.add_trace(go.Scatter(x=slot_df['SlotLabel'], y=slot_df['FTY'], mode='lines+markers', name='FTY', line=dict(color=G)))
        fig.add_trace(go.Scatter(x=slot_df['SlotLabel'], y=slot_df['UCL'], line=dict(color=R, dash='dash'), name='UCL'))
        fig.add_trace(go.Scatter(x=slot_df['SlotLabel'], y=slot_df['LCL'], line=dict(color=R, dash='dash'), name='LCL', fill='tonexty'))
        fig.update_layout(height=350, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="shg">🕒 Shift Analytics (Fixed Classification: Night 22-6, Morning 6-14, Afternoon 14-22)</div>', unsafe_allow_html=True)
    if 'Shift' in df.columns:
        shift_stats = df.groupby('Shift').apply(lambda g: pd.Series({
            'FPY': (g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100 if g['Prime_Handle'].sum()>0 else np.nan,
            'Cpk': cpk(g['FTY'].dropna()),
            'Handles': g['Prime_Handle'].sum()
        })).reset_index()
        fig_s = px.bar(shift_stats, x='Shift', y='FPY', color='Shift', text=shift_stats['FPY'].round(2).astype(str)+'%')
        fig_s.update_traces(textposition='outside')
        st.plotly_chart(fig_s, use_container_width=True)
        # ANOVA
        shift_groups = [df[df['Shift']==s]['FTY'].dropna() for s in shift_stats['Shift'] if len(df[df['Shift']==s]['FTY'].dropna())>1]
        if len(shift_groups) >= 2:
            f_val, p_val = stats.f_oneway(*shift_groups)
            if p_val < 0.05:
                st.info(f"📊 **Statistical difference between shifts detected** (p={p_val:.4f}). Review shift-specific practices.")
            else:
                st.success("No significant shift-to-shift variation detected.")

# --------------------- TAB 1: Process·Line ---------------------
with tabs[1]:
    proc = df.groupby('Process').apply(lambda g: pd.Series({
        'FPY': (g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100 if g['Prime_Handle'].sum()>0 else np.nan,
        'FTY': g['Prime_Pass'].sum()/g['Prime_Handle'].sum()*100 if g['Prime_Handle'].sum()>0 else np.nan,
        'Cpk': cpk(g['FTY'].dropna()),
        'DPMO': (g['Prime_Fail'].sum()/g['Prime_Handle'].sum())*1_000_000 if g['Prime_Handle'].sum()>0 else 0,
        'Handles': int(g['Prime_Handle'].sum())
    })).reset_index().dropna()
    fig_proc = go.Figure()
    fig_proc.add_trace(go.Bar(x=proc['Process'], y=proc['FPY'], name='FPY', marker_color=B, text=proc['FPY'].round(2).astype(str)+'%'))
    fig_proc.add_trace(go.Bar(x=proc['Process'], y=proc['FTY'], name='FTY', marker_color=G))
    fig_proc.update_layout(barmode='group', title='Process Yields')
    st.plotly_chart(fig_proc, use_container_width=True)

# --------------------- TAB 2: Tech·Family ---------------------
with tabs[2]:
    tech = df.groupby('Technology').apply(lambda g: pd.Series({
        'FPY': (g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100 if g['Prime_Handle'].sum()>0 else np.nan,
        'HFI': g['HFI'].mean(),
        'Handles': g['Prime_Handle'].sum()
    })).reset_index().dropna()
    fig_tech = px.bar(tech, x='Technology', y='FPY', color='HFI', text=tech['FPY'].round(2).astype(str)+'%',
                      color_continuous_scale='OrRd', title='FPY by Technology (color = Hidden Factory %)')
    st.plotly_chart(fig_tech, use_container_width=True)

# --------------------- TAB 3: Monte Carlo ---------------------
with tabs[3]:
    fpy_m = df['FPY'].dropna().mean(); fpy_s = df['FPY'].dropna().std()
    sims = np.random.normal(fpy_m, max(fpy_s,0.01), (4000,30))
    sims = np.clip(sims, 0, 100)
    pct = {f'p{p}': np.percentile(sims, p, axis=0) for p in [5,25,50,75,95]}
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(x=np.arange(1,31), y=pct['p95'], fill=None))
    fig_mc.add_trace(go.Scatter(x=np.arange(1,31), y=pct['p5'], fill='tonexty', name='90% band'))
    fig_mc.add_trace(go.Scatter(x=np.arange(1,31), y=pct['p50'], name='Median', line=dict(color=B, width=2.5)))
    fig_mc.add_hline(y=tfpy, line_dash='dash', line_color=R)
    st.plotly_chart(fig_mc, use_container_width=True)

# --------------------- TAB 4: Bayesian ------------------------
with tabs[4]:
    def bayesian(fails, handles, a0=1, b0=19):
        pa = a0+fails; pb = b0+(handles-fails)
        x = np.linspace(0, 0.25, 300)
        prior = stats.beta.pdf(x, a0, b0)
        post = stats.beta.pdf(x, pa, pb)
        pm = pa/(pa+pb)
        ci = stats.beta.ppf([0.025,0.975], pa, pb)
        return x, prior, post, pm, ci[0], ci[1]
    x, prior, post, pm, cl, ch = bayesian(K['PF'], K['PH'])
    fig_bay = go.Figure()
    fig_bay.add_trace(go.Scatter(x=x*100, y=prior/100, fill='tozeroy', name='Prior', line=dict(dash='dash')))
    fig_bay.add_trace(go.Scatter(x=x*100, y=post/100, fill='tozeroy', name='Posterior', line=dict(color=B)))
    fig_bay.add_vline(x=pm*100, line_dash='dash', line_color=R, annotation_text=f"μ={pm*100:.2f}%")
    st.plotly_chart(fig_bay, use_container_width=True)

# --------------------- TAB 5: Markov --------------------------
with tabs[5]:
    def markov(df, good, warn):
        def stt(v):
            if pd.isna(v): return None
            return 'Good' if v>=good else ('Warning' if v>=warn else 'Fail')
        sl_fpy = df.groupby('SlotLabel').apply(lambda g: (g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100 if g['Prime_Handle'].sum()>0 else np.nan).reset_index()
        sl_fpy.columns = ['SlotLabel','FPY']
        states = sl_fpy['FPY'].apply(stt).dropna().tolist()
        cats = ['Good','Warning','Fail']
        T = pd.DataFrame(0, index=cats, columns=cats)
        for a,b in zip(states[:-1], states[1:]):
            if a in cats and b in cats: T.loc[a,b] += 1
        rowsums = T.sum(axis=1)
        T = T.div(rowsums.where(rowsums>0,1), axis=0)
        return T
    TM = markov(df, tfpy, tfpy-5)
    fig_mark = px.imshow(TM.round(3), text_auto='.3f', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_mark, use_container_width=True)

# --------------------- TAB 6: SPC·Cpk·Ppk ---------------------
with tabs[6]:
    st.markdown('<div class="sh">📊 Process Capability over Time</div>', unsafe_allow_html=True)
    slot_fty = df.groupby('SlotLabel')['FTY'].mean().reset_index()
    if len(slot_fty) > 5:
        window = min(10, len(slot_fty)//2)
        slot_fty['Cpk_rolling'] = slot_fty['FTY'].rolling(window, min_periods=5).apply(lambda x: cpk(x) if len(x)>=2 else np.nan)
        slot_fty['Ppk_rolling'] = slot_fty['FTY'].rolling(window, min_periods=5).apply(lambda x: ppk(x) if len(x)>=2 else np.nan)
        fig_cpk = go.Figure()
        fig_cpk.add_trace(go.Scatter(x=slot_fty['SlotLabel'], y=slot_fty['Cpk_rolling'], name='Cpk (within)', line=dict(color=B)))
        fig_cpk.add_trace(go.Scatter(x=slot_fty['SlotLabel'], y=slot_fty['Ppk_rolling'], name='Ppk (overall)', line=dict(color=O, dash='dot')))
        fig_cpk.add_hline(y=target_cpk, line_dash='dash', line_color=R)
        st.plotly_chart(fig_cpk, use_container_width=True)

# --------------------- TAB 7: Anomaly Clusters + Root Cause Summary -------------
with tabs[7]:
    if NA > 5:
        anom = df[df['Anom']].copy()
        feat = ['FTY','FailPct','NTFPct']
        X_an = StandardScaler().fit_transform(anom[feat].fillna(0))
        kmeans = KMeans(n_clusters=min(3, len(anom)), random_state=42, n_init=10)
        anom['Cluster'] = kmeans.fit_predict(X_an).astype(str)
        fig_cl = px.scatter(anom, x='FTY', y='FailPct', color='Cluster', size='Prime_Handle', hover_data=['Line','Process'])
        st.plotly_chart(fig_cl, use_container_width=True)
        st.markdown("#### 🔍 Root Cause Summary")
        for cl in sorted(anom['Cluster'].unique()):
            sub = anom[anom['Cluster']==cl]
            avg_fail = sub['FailPct'].mean()
            avg_fty = sub['FTY'].mean()
            if avg_fail > 10:
                st.markdown(f"- **Cluster {cl}**: High failure rate ({avg_fail:.1f}%) → Likely process/material issue. Review raw material or machine parameters.")
            elif avg_fty < tfty:
                st.markdown(f"- **Cluster {cl}**: Low FTY ({avg_fty:.1f}%) → Systemic capability problem. Invest in operator training or tooling.")
            else:
                st.markdown(f"- **Cluster {cl}**: Mixed signature – inspect specific lines/slots: {sub['Line'].unique()[:3]}")
    else:
        st.info("Not enough anomalies to cluster (need >5).")

# --------------------- TAB 8: Pareto·RTY -----------------------
with tabs[8]:
    proc_loss = df.groupby('Process').apply(lambda g: pd.Series({
        'YieldLoss': 100 - ((g['Prime_Pass'].sum()+g['PrimeCount'].sum())/g['Prime_Handle'].sum()*100) if g['Prime_Handle'].sum()>0 else 0,
        'HFI': g['HFI'].mean()
    })).reset_index()
    proc_loss['WeightedLoss'] = proc_loss['YieldLoss'] * (proc_loss['HFI']/100)
    proc_loss = proc_loss[proc_loss['WeightedLoss']>0].sort_values('WeightedLoss', ascending=False).head(10)
    fig_pareto = px.bar(proc_loss, x='Process', y='WeightedLoss', color='HFI', text=proc_loss['WeightedLoss'].round(2).astype(str))
    st.plotly_chart(fig_pareto, use_container_width=True)
    if not rty_df.empty:
        fig_rty = go.Figure()
        fig_rty.add_trace(go.Bar(x=rty_df['Process'], y=rty_df['FPY_pct'], marker_color=B, text=rty_df['FPY_pct'].round(2).astype(str)+'%'))
        fig_rty.add_trace(go.Scatter(x=rty_df['Process'], y=rty_df['RTY_run'], mode='lines+markers', name='Cumulative RTY', line=dict(color=R, width=2.5)))
        fig_rty.update_layout(title=f'RTY = {OV_RTY:.2f}%')
        st.plotly_chart(fig_rty, use_container_width=True)

# --------------------- TAB 9: OEE·TEEP + Loss Tree ------------------------
with tabs[9]:
    st.markdown('<div class="sh">🏭 OEE Loss Tree</div>', unsafe_allow_html=True)
    loss_data = pd.DataFrame({
        'Loss Category': ['Planned Downtime', 'Speed Loss', 'Quality Loss'],
        'Loss %': [(1-availability)*100, (1-performance)*100, (1-quality)*100]
    })
    fig_loss = px.bar(loss_data, x='Loss Category', y='Loss %', color='Loss Category', text=loss_data['Loss %'].round(1).astype(str)+'%')
    fig_loss.update_layout(title='OEE Loss Breakdown')
    st.plotly_chart(fig_loss, use_container_width=True)
    st.metric("OEE", f"{OEE*100:.1f}%", delta="World class ≥85%" if OEE>=0.85 else "Improve")
    st.metric("TEEP", f"{TEEP*100:.1f}%")
    # Muda/Mura/Muri
    st.markdown('<div class="shg">⚠️ Lean Wastes (Muda, Mura, Muri)</div>', unsafe_allow_html=True)
    muda = K['HFI']
    mura = VOL
    muri = max(0, (K['PH'] / (planned_prod_time * 60 / ideal_cycle_time) - 1) * 100) if ideal_cycle_time>0 else 0
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Muda (Waste %)", f"{muda:.1f}%", help="Hidden Factory = rework")
    col_b.metric("Mura (Inconsistency %)", f"{mura:.1f}%", help="Coefficient of variation of FPY")
    col_c.metric("Muri (Overburden %)", f"{muri:.1f}%", help="Excess load vs designed capacity")

# --------------------- TAB 10: FMEA·CAPA + Quick Wins -----------------------
with tabs[10]:
    st.markdown('<div class="sh">📋 FMEA with RPN Alerts</div>', unsafe_allow_html=True)
    top_fail = df.groupby('Process')['Prime_Fail'].sum().reset_index().sort_values('Prime_Fail', ascending=False).head(5)
    fmea_rows = []
    for _, row in top_fail.iterrows():
        sev = 7 if row['Prime_Fail'] > K['PF']/len(df) else 4
        occ = min(10, int(row['Prime_Fail'] / max(1, K['PH']/len(df)) * 10))
        det = 5
        rpn = sev * occ * det
        fmea_rows.append({'Process':row['Process'], 'Failure Mode':'Yield loss', 'RPN':rpn, 'Action':'Optimize parameters'})
    fmea_df = pd.DataFrame(fmea_rows)
    st.dataframe(fmea_df, use_container_width=True)
    # Highlight RPN > 100
    high_rpn = fmea_df[fmea_df['RPN'] > 100]
    if not high_rpn.empty:
        st.error(f"⚠️ High RPN processes: {', '.join(high_rpn['Process'])} – prioritize corrective actions.")
    st.markdown('<div class="shg">🚀 Quick Wins (Based on Current Data)</div>', unsafe_allow_html=True)
    quick_wins = []
    if K['HFI'] > 20:
        quick_wins.append("Reduce Hidden Factory: target rework loops (NTF).")
    if CPK < 1.0:
        quick_wins.append(f"Improve centering: shift FTY mean towards {tfty}%.")
    if VOL > 15:
        quick_wins.append("Stabilise process: high volatility indicates inconsistent inputs.")
    if NA > 10:
        quick_wins.append("Investigate top anomaly clusters – see Anomaly Clusters tab.")
    if not quick_wins:
        quick_wins.append("Process appears stable. Maintain with SPC monitoring.")
    for w in quick_wins:
        st.markdown(f"- {w}")

# --------------------- TAB 11: COPQ·DPMO --------------------
with tabs[11]:
    st.markdown('<div class="sh">💰 Cost of Poor Quality (₹)</div>', unsafe_allow_html=True)
    copq_data = pd.DataFrame({
        'Category': ['Internal Failure (rework)', 'Internal Failure (scrap)', 'External Failure'],
        'Cost': [rework_cost, scrap_cost, external_cost]
    })
    fig_copq = px.pie(copq_data, values='Cost', names='Category', title='COPQ Breakdown')
    st.plotly_chart(fig_copq, use_container_width=True)
    st.metric("Total COPQ per batch", f"₹{coq:,.0f}")
    st.metric("DPMO", f"{K['DPMO']:,}", delta=f"Sigma Level = {K['Sigma']:.1f}")
    if K['Sigma'] < 3:
        st.warning("Sigma level below 3 – urgent improvement needed.")

# --------------------- TAB 12: Raw Data -----------------------
with tabs[12]:
    tbl = st.radio("Table", ["rld1", "rld1_hidden"], horizontal=True)
    if tbl == "rld1":
        disp_cols = ['Date','SlotLabel','Shift','Line','Process','Technology','Family','FPY','FTY','HFI','DPMO','FailPct','NTFPct','Prime_Handle']
        st.dataframe(df[disp_cols].sort_values(['Date','SlotLabel']), use_container_width=True, height=500)
    else:
        st.dataframe(hf, use_container_width=True, height=500)
    st.download_button("📥 Download CSV", df.to_csv(index=False), f"yield_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")

st.markdown("---")
st.caption(f"🔄 {datetime.now():%Y-%m-%d %H:%M:%S} | Industrial Excellence v5.1 | Shift fixed | ₹ currency | Centering corrected | COPQ fixed")
