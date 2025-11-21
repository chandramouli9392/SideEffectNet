# dashboard.py (FINAL — Full app with floating pills background)
import os
from pathlib import Path
import tempfile

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from dotenv import load_dotenv
from google import genai

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="SideEffectNet Dashboard",
    layout="wide",
    page_icon="media/sideeffectnetlogo.png",
    menu_items={
        'Get Help': 'https://github.com/ganeshmysoreDT/SideEffectNet',
        'Report a bug': "https://github.com/ganeshmysoreDT/SideEffectNet/issues",
        'About': "# SideEffectNet: Drug Safety Analytics Platform"
    }
)

# -----------------------
# Environment / API keys
# -----------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -----------------------
# Data files
# -----------------------
DATA_DIR = Path("data/processed")
EDGE_CSV = DATA_DIR / "side_effects_clean.csv"
RISK_CSV = DATA_DIR / "drug_risk_scores.csv"

# -----------------------
# Data loading & graph utilities
# -----------------------
@st.cache_data(show_spinner="Loading data...")
def load_data():
    # Defensive: if files missing, return empty frames
    if not EDGE_CSV.exists() or not RISK_CSV.exists():
        return pd.DataFrame(columns=["drug_name", "side_effect", "freq_pct"]), pd.DataFrame(columns=["drug_name", "risk_score"])
    edges = pd.read_csv(EDGE_CSV, usecols=["drug_name", "side_effect", "freq_pct"])
    risks = pd.read_csv(RISK_CSV, usecols=["drug_name", "risk_score"])
    return edges, risks

@st.cache_data(show_spinner="Building graph...")
def build_graph(edges_df):
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        drug = str(row["drug_name"]).strip()
        se = str(row["side_effect"]).strip()
        freq_raw = row.get("freq_pct", None)
        try:
            freq = float(freq_raw)
        except (ValueError, TypeError):
            freq = "N/A"
        G.add_node(drug, type="drug", color="#636EFA", size=20)
        G.add_node(se, type="side_effect", color="#EF553B", size=15)
        G.add_edge(drug, se, frequency=freq, title=f"Frequency: {freq}%")
    return G

@st.cache_data(show_spinner="Computing centrality...")
def compute_centrality(_G):
    # k set for approximate centrality on larger graphs
    try:
        k = min(100, max(1, len(_G.nodes())))
        return nx.betweenness_centrality(_G, k=k)
    except Exception:
        return nx.betweenness_centrality(_G)

# -----------------------
# Load & prepare
# -----------------------
edges_df, risk_df = load_data()
G = build_graph(edges_df.head(500))

risk_map = {}
if not risk_df.empty:
    risk_map = risk_df.set_index("drug_name")["risk_score"].to_dict()
side_effect_lookup = {drug: list(group["side_effect"]) for drug, group in edges_df.groupby("drug_name")}

# -----------------------
# Sidebar - UI Settings
# -----------------------
with st.sidebar:
    st.markdown("## UI Settings")
    theme_choice = st.radio("Theme", options=["Auto", "Light", "Dark"], index=0,
                            help="Auto follows your OS / browser preference")
    reduce_motion = st.checkbox("Reduce animation (accessibility)", value=False,
                               help="Disable floating background animation for accessibility")

# Convert reduce_motion into a CSS snippet safely (no f-string)
if reduce_motion:
    reduce_motion_css = "animation: none !important; opacity: 0.95;"
else:
    reduce_motion_css = ""  # empty => animation runs

# -----------------------
# CSS + Floating pills background (no f-strings, placeholder replaced)
# -----------------------
css_template = """
<style>
/* Make Streamlit root containers transparent so background shows through */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > .main > div,
.stApp, .block-container {
    background: transparent !important;
    background-color: transparent !important;
}

/* Fullscreen background layer for animated pills */
#sideeffectnet-bg {
    pointer-events: none; /* so clicks pass through */
    position: fixed;
    inset: 0;
    z-index: -9999;
    overflow: hidden;
    display: block;
}

/* Gradient base that is visible under pills (auto adapts to prefers-color-scheme) */
:root {
  --grad-light: linear-gradient(120deg, rgba(126, 92, 194, 0.85) 0%, rgba(91, 170, 255, 0.85) 40%, rgba(240, 120, 145, 0.85) 100%);
  --grad-dark: linear-gradient(120deg, #0b1220 0%, #0f172a 30%, #1f2937 70%);
}

/* prefer dark */
@media (prefers-color-scheme: dark) {
  #sideeffectnet-bg::before {
      content: "";
      position: absolute;
      inset: 0;
      background: var(--grad-dark);
      background-size: 300% 300%;
      filter: saturate(1.02) contrast(1.02) blur(0px);
      opacity: 0.95;
      transform: translateZ(0);
      animation: bgShiftDark 18s linear infinite;
  }
}

@media (prefers-color-scheme: light) {
  #sideeffectnet-bg::before {
      content: "";
      position: absolute;
      inset: 0;
      background: var(--grad-light);
      background-size: 300% 300%;
      filter: saturate(1.05) contrast(1.03) blur(0px);
      opacity: 0.95;
      transform: translateZ(0);
      animation: bgShiftLight 14s linear infinite;
  }
}

/* If you later want to force via classes, add overrides here (not used by default) */
html.sideeffectnet-theme-light #sideeffectnet-bg::before {
    content: "";
    position: absolute;
    inset: 0;
    background: var(--grad-light);
    background-size: 300% 300%;
    animation: bgShiftLight 14s linear infinite;
}
html.sideeffectnet-theme-dark #sideeffectnet-bg::before {
    content: "";
    position: absolute;
    inset: 0;
    background: var(--grad-dark);
    background-size: 300% 300%;
    animation: bgShiftDark 18s linear infinite;
}

@keyframes bgShiftLight {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes bgShiftDark {
  0% { background-position: 20% 50%; }
  50% { background-position: 80% 50%; }
  100% { background-position: 20% 50%; }
}

/* Container for pills */
#sideeffectnet-bg .pill {
    position: absolute;
    width: 72px;
    height: 34px;
    border-radius: 24px;
    background: linear-gradient(90deg, rgba(255,255,255,0.95), rgba(230,230,255,0.9));
    box-shadow: 0 8px 20px rgba(4,6,20,0.18), inset 0 -4px 8px rgba(0,0,0,0.08);
    transform-origin: center;
    opacity: 0.9;
    display: block;
    will-change: transform, top, left, opacity;
    __REDUCE_MOTION_CSS__
}

/* small pill variant */
#sideeffectnet-bg .pill.small {
    width: 44px;
    height: 22px;
    border-radius: 16px;
    opacity: 0.85;
    transform-origin: center;
}

/* pill inner stripe to make it look like tablets/capsules */
#sideeffectnet-bg .pill::after {
    content: "";
    position: absolute;
    left: 10%;
    right: 10%;
    top: 10%;
    bottom: 10%;
    border-radius: 12px;
    background: linear-gradient(90deg, rgba(255,0,120,0.12), rgba(0,160,255,0.12));
    mix-blend-mode: overlay;
    pointer-events: none;
}

/* pill label / AI icon (small) */
#sideeffectnet-bg .pill .icon {
    position: absolute;
    right: 10px;
    top: 6px;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.9), rgba(255,255,255,0.15));
    box-shadow: 0 2px 6px rgba(2,6,23,0.2);
    opacity: 0.95;
    pointer-events: none;
}

/* animation: pills float and drift upwards with gentle rotation and side-to-side */
@keyframes floatUp {
  0% {
    transform: translate3d(0, 0vh, 0) rotate(0deg);
    opacity: 0;
  }
  10% {
    opacity: 0.95;
  }
  50% {
    transform: translate3d(30px, -40vh, 0) rotate(10deg);
    opacity: 0.95;
  }
  100% {
    transform: translate3d(-20px, -110vh, 0) rotate(-8deg);
    opacity: 0;
  }
}

@keyframes floatUpSlow {
  0% {
    transform: translate3d(0, 0vh, 0) rotate(0deg);
    opacity: 0;
  }
  10% {
    opacity: 0.9;
  }
  50% {
    transform: translate3d(-40px, -35vh, 0) rotate(-6deg);
    opacity: 0.95;
  }
  100% {
    transform: translate3d(20px, -110vh, 0) rotate(6deg);
    opacity: 0;
  }
}

/* We create many pill instances and vary their delays/durations/left positions via nth-child selectors */
#sideeffectnet-bg .pill:nth-child(1) {
    left: 8%;
    bottom: -10%;
    animation: floatUp 18s linear infinite;
    animation-delay: 0s;
    transform: scale(1.05) rotate(6deg);
}
#sideeffectnet-bg .pill:nth-child(2) {
    left: 22%;
    bottom: -12%;
    animation: floatUpSlow 22s linear infinite;
    animation-delay: 2s;
    transform: scale(0.9) rotate(-4deg);
}
#sideeffectnet-bg .pill:nth-child(3) {
    left: 36%;
    bottom: -8%;
    animation: floatUp 20s linear infinite;
    animation-delay: 5s;
    transform: scale(1.0) rotate(2deg);
}
#sideeffectnet-bg .pill:nth-child(4) {
    left: 50%;
    bottom: -14%;
    animation: floatUpSlow 24s linear infinite;
    animation-delay: 1s;
    transform: scale(1.1) rotate(10deg);
}
#sideeffectnet-bg .pill:nth-child(5) {
    left: 64%;
    bottom: -9%;
    animation: floatUp 19s linear infinite;
    animation-delay: 4s;
    transform: scale(0.95) rotate(-8deg);
}
#sideeffectnet-bg .pill:nth-child(6) {
    left: 78%;
    bottom: -11%;
    animation: floatUpSlow 21s linear infinite;
    animation-delay: 6s;
    transform: scale(0.85) rotate(4deg);
}
#sideeffectnet-bg .pill:nth-child(7) {
    left: 12%;
    bottom: -20%;
    animation: floatUp 25s linear infinite;
    animation-delay: 8s;
    transform: scale(0.8) rotate(-6deg);
}
#sideeffectnet-bg .pill:nth-child(8) {
    left: 30%;
    bottom: -20%;
    animation: floatUpSlow 23s linear infinite;
    animation-delay: 10s;
    transform: scale(1.15) rotate(14deg);
}
#sideeffectnet-bg .pill:nth-child(9) {
    left: 46%;
    bottom: -18%;
    animation: floatUp 17s linear infinite;
    animation-delay: 3s;
    transform: scale(0.9) rotate(3deg);
}
#sideeffectnet-bg .pill:nth-child(10) {
    left: 62%;
    bottom: -22%;
    animation: floatUpSlow 26s linear infinite;
    animation-delay: 9s;
    transform: scale(1.05) rotate(-12deg);
}
#sideeffectnet-bg .pill:nth-child(11) {
    left: 82%;
    bottom: -19%;
    animation: floatUp 27s linear infinite;
    animation-delay: 12s;
    transform: scale(0.95) rotate(6deg);
}
#sideeffectnet-bg .pill:nth-child(12) {
    left: 4%;
    bottom: -25%;
    animation: floatUpSlow 28s linear infinite;
    animation-delay: 7s;
    transform: scale(1.0) rotate(-10deg);
}
#sideeffectnet-bg .pill:nth-child(13) {
    left: 18%;
    bottom: -24%;
    animation: floatUp 22s linear infinite;
    animation-delay: 14s;
    transform: scale(1.07) rotate(8deg);
}
#sideeffectnet-bg .pill:nth-child(14) {
    left: 34%;
    bottom: -21%;
    animation: floatUpSlow 20s linear infinite;
    animation-delay: 11s;
    transform: scale(0.92) rotate(-5deg);
}
#sideeffectnet-bg .pill:nth-child(15) {
    left: 48%;
    bottom: -26%;
    animation: floatUp 30s linear infinite;
    animation-delay: 16s;
    transform: scale(1.12) rotate(10deg);
}
#sideeffectnet-bg .pill:nth-child(16) {
    left: 66%;
    bottom: -27%;
    animation: floatUpSlow 29s linear infinite;
    animation-delay: 13s;
    transform: scale(0.86) rotate(-7deg);
}
#sideeffectnet-bg .pill:nth-child(17) {
    left: 80%;
    bottom: -28%;
    animation: floatUp 24s linear infinite;
    animation-delay: 15s;
    transform: scale(1.0) rotate(4deg);
}
#sideeffectnet-bg .pill:nth-child(18) {
    left: 56%;
    bottom: -30%;
    animation: floatUpSlow 32s linear infinite;
    animation-delay: 18s;
    transform: scale(0.9) rotate(-4deg);
}
#sideeffectnet-bg .pill:nth-child(19) {
    left: 26%;
    bottom: -30%;
    animation: floatUp 34s linear infinite;
    animation-delay: 17s;
    transform: scale(0.78) rotate(-12deg);
}
#sideeffectnet-bg .pill:nth-child(20) {
    left: 72%;
    bottom: -32%;
    animation: floatUpSlow 33s linear infinite;
    animation-delay: 19s;
    transform: scale(1.18) rotate(12deg);
}

/* smaller pills layered in front */
#sideeffectnet-bg .pill.small:nth-child(21) { left: 10%; bottom: -8%; animation: floatUp 16s linear infinite; animation-delay: 2s; transform: scale(0.6); }
#sideeffectnet-bg .pill.small:nth-child(22) { left: 40%; bottom: -16%; animation: floatUpSlow 19s linear infinite; animation-delay: 5s; transform: scale(0.7); }
#sideeffectnet-bg .pill.small:nth-child(23) { left: 70%; bottom: -12%; animation: floatUp 18s linear infinite; animation-delay: 6s; transform: scale(0.65); }
#sideeffectnet-bg .pill.small:nth-child(24) { left: 52%; bottom: -6%; animation: floatUpSlow 17s linear infinite; animation-delay: 1s; transform: scale(0.55); }

/* ensure main UI sits above background */
[data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 0;
}
.block-container {
    position: relative;
    z-index: 10;
}

/* glass cards for contrast */
.stMarkdown, .stFrame, .stDataFrameContainer, .stDataFrame, .stAlert, .stMetric, .stButton {
    background: rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(8px) saturate(120%) !important;
    border-radius: 10px !important;
    padding: 0.6rem !important;
    color: inherit;
}

/* dark-mode tweaks for readability */
@media (prefers-color-scheme: dark) {
    .stMarkdown, .stFrame, .stDataFrameContainer, .stDataFrame, .stAlert, .stMetric, .stButton {
        background: rgba(5,8,12,0.55) !important;
        color: #e6eef8 !important;
    }
}

/* Responsive adjustments */
@media (max-width: 600px) {
    #sideeffectnet-bg .pill { width: 54px; height: 26px; }
    #sideeffectnet-bg .pill.small { width: 32px; height: 16px; }
}
</style>

<!-- background HTML: pill elements. No JS required. -->
<div id="sideeffectnet-bg" aria-hidden="true">
    <!-- 20 larger pills -->
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>
    <div class="pill"><div class="icon"></div></div>

    <!-- 4 smaller pills for depth -->
    <div class="pill small"><div class="icon"></div></div>
    <div class="pill small"><div class="icon"></div></div>
    <div class="pill small"><div class="icon"></div></div>
    <div class="pill small"><div class="icon"></div></div>
</div>
"""

# Replace placeholder with the actual reduce_motion_css value safely:
css_html = css_template.replace("__REDUCE_MOTION_CSS__", reduce_motion_css)

# Inject CSS + background HTML
st.markdown(css_html, unsafe_allow_html=True)

# -----------------------
# Sidebar content (logo + filters)
# -----------------------
with st.sidebar:
    # Put logo and filters after the UI settings at top
    try:
        st.image("media/sideeffectnetlogo.png", width=150)
    except Exception:
        # If image missing, show title fallback
        st.markdown("**SideEffectNet**")

    st.markdown("### Filters")
    if risk_df.empty:
        st.caption("No data loaded - check data/processed/ files.")
        min_risk, max_risk = 0.0, 1.0
        risk_filter = (0.0, 1.0)
    else:
        min_risk, max_risk = float(risk_df["risk_score"].min()), float(risk_df["risk_score"].max())
        risk_filter = st.slider(
            "Filter by risk score",
            min_value=min_risk,
            max_value=max_risk,
            value=(min_risk, max_risk)
        )
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    SideEffectNet is a drug safety analytics platform that:
    - Visualizes drug-side effect relationships
    - Calculates risk scores for medications
    - Identifies potential polypharmacy risks
    """)
    st.markdown("---")
    st.markdown("Data source: FDA Adverse Event Reporting System")

# -----------------------
# Main app content (tabs)
# -----------------------
st.title("SideEffectNet: Drug Safety Analytics")
st.markdown("Explore drug-side effect relationships, risk scores, and polypharmacy risks.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Drug Lookup",
    "Safer Alternatives",
    "Risk Explorer",
    "Polypharmacy",
    "Critical Nodes",
    "Risk Hypotheses"
])

# -----------------------
# TAB 1 - Drug Lookup
# -----------------------
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Drug Profile")
        drug_list = sorted(set(risk_map.keys()))
        drug = st.selectbox("Search for a drug:", options=[""] + drug_list, index=0, key="tab1_drug_search")
        if drug:
            if drug not in risk_map:
                st.error("Drug not found in dataset.")
            else:
                score = risk_map[drug]
                risk_color = "red" if score > 0.7 else "orange" if score > 0.4 else "green"
                st.markdown(f"""
<div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid {risk_color}">
    <h3 style="margin-top: 0; color: #24292f;">{drug}</h3>
    <div style="font-size: 2rem; font-weight: bold; color: {risk_color};">{score:.3f}</div>
    <div style="color: #57606a;">Risk Score (0-1 scale)</div>
</div>
""", unsafe_allow_html=True)
                se_list = list(dict.fromkeys(side_effect_lookup.get(drug, [])))
                st.markdown(f"#### Reported Side Effects ({len(se_list)} total)")
                if se_list:
                    for i, se in enumerate(se_list[:10], start=1):
                        st.markdown(f"- {se}")
                    if len(se_list) > 10:
                        with st.expander("Show all side effects"):
                            for i, se in enumerate(se_list[10:], start=11):
                                st.markdown(f"- {se}")
                else:
                    st.info("No side effects recorded for this drug.")

    with col2:
        if drug and drug in side_effect_lookup and st.checkbox("Show Network Visualization", True):
            se_list = list(dict.fromkeys(side_effect_lookup.get(drug, [])))
            if se_list:
                sg = nx.Graph()
                sg.add_node(drug, color="#636EFA", size=30, title=f"Drug: {drug}\\nRisk: {risk_map.get(drug, 'N/A')}")
                for se in se_list[:20]:
                    sg.add_node(se, color="#EF553B", size=20, title=f"Side Effect: {se}")
                    if G.has_edge(drug, se):
                        freq = G.edges[drug, se].get("frequency", "N/A")
                        sg.add_edge(drug, se, value=freq if isinstance(freq, (int, float)) else 1, title=f"Frequency: {freq}%")
                    else:
                        sg.add_edge(drug, se, value=1, title="Frequency: N/A")
                pv = Network(height="600px", width="100%", bgcolor="white", font_color="black", directed=False, notebook=False)
                pv.from_nx(sg)
                pv.set_options("""
{
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -80000,
      "centralGravity": 0.3,
      "springLength": 95,
      "springConstant": 0.04,
      "damping": 0.09,
      "avoidOverlap": 0.1
    },
    "minVelocity": 0.75,
    "solver": "barnesHut"
  },
  "nodes": {
    "borderWidth": 2,
    "borderWidthSelected": 3,
    "shape": "dot",
    "scaling": {
      "min": 10,
      "max": 30
    },
    "font": {
      "size": 14,
      "face": "arial"
    }
  },
  "edges": {
    "color": {
      "inherit": true
    },
    "smooth": {
      "enabled": true,
      "type": "continuous"
    },
    "width": 2
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 200,
    "hideEdgesOnDrag": true,
    "multiselect": true
  }
}
""")
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
                    pv.save_graph(tmpfile.name)
                    components.html(open(tmpfile.name, 'r').read(), height=620, scrolling=False)
            else:
                st.warning("No side effects to visualize for this drug.")

# -----------------------
# TAB 2 - Safer Alternatives
# -----------------------
with tab2:
    st.header("Find Safer Alternatives")
    st.markdown("Discover drugs with similar effects but lower risk profiles.")
    if 'drug' in locals() and drug and drug in side_effect_lookup:
        target_set = set(side_effect_lookup[drug])
        risk_query = risk_map[drug]
        with st.spinner("Analyzing alternatives..."):
            suggestions = []
            for other, se_list in side_effect_lookup.items():
                if other == drug:
                    continue
                overlap = len(target_set & set(se_list))
                if overlap == 0:
                    continue
                risk_other = risk_map.get(other, float("inf"))
                if risk_other < risk_query:
                    suggestions.append({
                        "Drug": other,
                        "Shared Effects": overlap,
                        "Risk Score": risk_other,
                        "Risk Reduction": risk_query - risk_other
                    })
            if suggestions:
                sugg_df = pd.DataFrame(suggestions).sort_values(["Shared Effects", "Risk Reduction"], ascending=[False, False])
                st.dataframe(sugg_df, use_container_width=True)
            else:
                st.info("No safer alternatives with overlapping side effects found.")
    else:
        st.info("Please select a drug in the 'Drug Lookup' tab first.")

# -----------------------
# TAB 3 - Risk Explorer
# -----------------------
with tab3:
    st.header("Drug Risk Explorer")
    st.markdown("Analyze and compare drug risk scores across the dataset.")
    if risk_df.empty:
        st.info("No risk data available.")
    filtered = risk_df[(risk_df["risk_score"] >= risk_filter[0]) & (risk_df["risk_score"] <= risk_filter[1])] if not risk_df.empty else pd.DataFrame(columns=["drug_name", "risk_score"])
    col1, col2 = st.columns([1, 3])
    with col1:
        drugs_in_range_color = "green" if len(filtered) > 50 else "orange" if len(filtered) > 20 else "red"
        st.markdown(f"""
        <div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid {drugs_in_range_color}; margin-bottom: 1rem;">
            <div style="font-size: 1rem; color: #57606a;">Drugs in Range</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {drugs_in_range_color};">{len(filtered)}</div>
        </div>
        """, unsafe_allow_html=True)
        if not filtered.empty:
            avg_risk = filtered["risk_score"].mean()
            avg_risk_color = "green" if avg_risk <= 0.4 else "orange" if avg_risk <= 0.7 else "red"
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid {avg_risk_color}; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Average Risk</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {avg_risk_color};">{avg_risk:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**Highest Risk in Range**")
            for _, row in filtered.nlargest(5, "risk_score").iterrows():
                st.markdown(f"- {row['drug_name']} ({row['risk_score']:.3f})")
    with col2:
        if not filtered.empty:
            fig = px.histogram(filtered, x="risk_score", nbins=30, title="Distribution of Risk Scores", labels={"risk_score": "Risk Score"}, color_discrete_sequence=['#636EFA'])
            fig.update_layout(bargap=0.1, yaxis_title="Number of Drugs", xaxis_range=[risk_filter[0]-0.05, risk_filter[1]+0.05])
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Drug Risk Data")
    st.dataframe(filtered, use_container_width=True)

# -----------------------
# TAB 4 - Polypharmacy Risk Detection
# -----------------------
with tab4:
    st.header("Polypharmacy Risk Analyzer")
    st.markdown("Identify potential risks when combining multiple medications.")
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="width: 20px; height: 20px; background-color: green; margin-right: 10px; border-radius: 50%;"></div>
        <span style="font-size: 1rem; color: #57606a;">Safe</span>
    </div>
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="width: 20px; height: 20px; background-color: orange; margin-right: 10px; border-radius: 50%;"></div>
        <span style="font-size: 1rem; color: #57606a;">Moderate Risk</span>
    </div>
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="width: 20px; height: 20px; background-color: red; margin-right: 10px; border-radius: 50%;"></div>
        <span style="font-size: 1rem; color: #57606a;">High Risk</span>
    </div>
    """, unsafe_allow_html=True)
    drug_options = sorted(risk_df["drug_name"].unique()) if not risk_df.empty else []
    selected_drugs = st.multiselect("Select 2-5 drugs to analyze combinations", drug_options, max_selections=5, help="Select multiple drugs to check for overlapping side effects")
    if len(selected_drugs) >= 2:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            combined_effects = set()
            overlap_effects = None
            combined_score = 0
            for d in selected_drugs:
                se_set = set(side_effect_lookup.get(d, []))
                combined_effects |= se_set
                overlap_effects = se_set if overlap_effects is None else overlap_effects & se_set
                combined_score += risk_map.get(d, 0)
            avg_score = combined_score / len(selected_drugs)
            max_score = max(risk_map.get(d, 0) for d in selected_drugs)
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid blue; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Average Risk</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: blue;">{avg_score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid red; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Highest Individual Risk</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: red;">{max_score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid green; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Total Unique Side Effects</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: green;">{len(combined_effects)}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid orange; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Overlapping Side Effects</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: orange;">{len(overlap_effects) if overlap_effects else 0}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            risk_level = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.4 else "Low"
            color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid {color}; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Combined Risk Level</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)
            if len(selected_drugs) > 2:
                st.warning("Combining more than 2 drugs increases risk exponentially")
        tab1a, tab1b = st.tabs(["Side Effect Overlap", "Risk Comparison"])
        with tab1a:
            if overlap_effects:
                st.markdown("### Overlapping Side Effects")
                for i, effect in enumerate(list(overlap_effects)[:20], start=1):
                    st.markdown(f"- {effect}")
                if len(overlap_effects) > 20:
                    st.markdown(f"... and {len(overlap_effects)-20} more")
            else:
                st.success("No overlapping side effects detected among selected drugs.")
        with tab1b:
            fig = go.Figure()
            for d in selected_drugs:
                fig.add_trace(go.Scatterpolar(r=[risk_map[d]], theta=[d], fill='toself', name=d))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Individual Drug Risk Scores")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least 2 drugs to analyze combinations")

# -----------------------
# TAB 5 - Critical Nodes (Centrality)
# -----------------------
with tab5:
    st.header("Network Critical Nodes Analysis")
    st.markdown("Identify the most influential drugs and side effects in the network.")
    if "centrality" not in st.session_state:
        with st.spinner("Computing network centrality..."):
            st.session_state["centrality"] = compute_centrality(G)
    centrality = st.session_state["centrality"]
    cent_df = pd.DataFrame([{"Node": n, "Type": G.nodes[n]["type"], "Centrality": c} for n, c in centrality.items()])
    drug_top = cent_df[cent_df["Type"] == "drug"].nlargest(10, "Centrality")
    se_top = cent_df[cent_df["Type"] == "side_effect"].nlargest(10, "Centrality")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Most Central Drugs")
        st.markdown("Drugs that connect to many side effects in the network")
        fig = px.bar(drug_top, x="Centrality", y="Node", orientation='h', color="Centrality", color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Clinical Implications**")
        st.markdown("- High centrality drugs may have broad side effect profiles\n- Potential for more drug-drug interactions\n- May require closer monitoring in clinical practice")
    with col2:
        st.markdown("### Most Central Side Effects")
        fig2 = px.bar(se_top, x="Centrality", y="Node", orientation='h', color="Centrality", color_continuous_scale='Plasma')
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**Clinical Implications**")
        st.markdown("- Common side effects across drug classes\n- May represent general physiological responses\n- Potential targets for preventative therapies")
    if st.checkbox("Show Critical Nodes Network", key="critical_nodes_network_checkbox"):
        top_nodes = list(drug_top["Node"]) + list(se_top["Node"])
        subgraph = G.subgraph(top_nodes)
        pv = Network(height="700px", width="100%", bgcolor="white", font_color="black", directed=False, notebook=False)
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node]["type"]
            pv.add_node(node, color="#636EFA" if node_type == "drug" else "#EF553B", size=30 if node_type == "drug" else 20, title=f"{node_type.capitalize()}: {node}\\nCentrality: {centrality[node]:.4f}", shape="dot")
        for edge in subgraph.edges():
            pv.add_edge(edge[0], edge[1], color="#A3A3A3", width=2, title=f"Edge between {edge[0]} and {edge[1]}")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
            pv.save_graph(tmpfile.name)
            components.html(open(tmpfile.name, 'r').read(), height=720, scrolling=False)

# -----------------------
# TAB 6 - Risk Hypotheses (Gemini)
# -----------------------
with tab6:
    st.header("AI-Powered Risk Hypotheses")
    st.markdown("Generate scientifically validated hypotheses about drug combination risks.")
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="width: 20px; height: 20px; background-color: green; margin-right: 10px; border-radius: 50%;"></div>
        <span style="font-size: 1rem; color: #57606a;">Safe</span>
    </div>
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="width: 20px; height: 20px; background-color: orange; margin-right: 10px; border-radius: 50%;"></div>
        <span style="font-size: 1rem; color: #57606a;">Moderate Risk</span>
    </div>
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="width: 20px; height: 20px; background-color: red; margin-right: 10px; border-radius: 50%;"></div>
        <span style="font-size: 1rem; color: #57606a;">High Risk</span>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("1. Select Drug Combination")
    col1, col2 = st.columns(2)
    with col1:
        drug_a = st.selectbox("Primary Drug", options=[""] + sorted(risk_map.keys()), index=0, key="tab6_primary_drug")
    with col2:
        drug_b = st.selectbox("Combination Drug", options=[""] + sorted(risk_map.keys()), index=0, key="tab6_secondary_drug")

    if drug_a and drug_b:
        st.subheader("2. Safety Analysis")
        risk_a = risk_map.get(drug_a, 0)
        risk_b = risk_map.get(drug_b, 0)
        side_effects_a = set(side_effect_lookup.get(drug_a, []))
        side_effects_b = set(side_effect_lookup.get(drug_b, []))
        overlapping = side_effects_a & side_effects_b

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            risk_a_color = "red" if risk_a > 0.7 else "orange" if risk_a > 0.4 else "green"
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid {risk_a_color}; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">{drug_a} Risk Score</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {risk_a_color};">{risk_a:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with metrics_col2:
            risk_b_color = "red" if risk_b > 0.7 else "orange" if risk_b > 0.4 else "green"
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid {risk_b_color}; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">{drug_b} Risk Score</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {risk_b_color};">{risk_b:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with metrics_col3:
            overlap_color = "red" if len(overlapping) > 10 else "orange" if len(overlapping) > 5 else "green"
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: rgba(255,255,255,0.85); border-left: 0.3rem solid {overlap_color}; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Shared Side Effects</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {overlap_color};">{len(overlapping)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("3. AI-Generated Hypotheses")
        if st.button("Generate Scientific Hypotheses", type="primary"):
            with st.spinner("Analyzing pharmacological profiles..."):
                context = {
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "risk_a": risk_a,
                    "risk_b": risk_b,
                    "overlap_count": len(overlapping),
                    "overlap_effects": list(overlapping)[:10]
                }

                # Build prompt string safely (simple f-string here for readability)
                prompt_template = f"""
As a senior pharmacologist, analyze this drug combination:

**Drugs**: {drug_a} (Risk: {risk_a:.2f}) + {drug_b} (Risk: {risk_b:.2f})

**Shared Side Effects**: {len(overlapping)}
**Key Overlaps**: {list(overlapping)[:10]}

Generate 3 clinically-relevant hypotheses considering:
1. Pharmacodynamic interactions
2. Metabolic pathway conflicts (CYP450, etc.)
3. Synergistic/adverse effect probabilities

For each hypothesis, provide:
- Mechanism of Action
- Biological Plausibility (1-5)
- Clinical Significance (High/Medium/Low)
- Suggested Monitoring Protocol
"""
                try:
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt_template
                    )
                    formatted_response = getattr(response, "text", str(response))
                    st.markdown("### Generated Hypotheses")
                    st.markdown(formatted_response)
                except Exception as e:
                    st.error(f"AI generation failed: {e}")

    else:
        st.info("Please select two drugs to analyze")

# End of file
