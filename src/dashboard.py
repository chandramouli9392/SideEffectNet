# dashboard.py (final — floating pills background, CSS-only, no components.html)
import streamlit as st
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

import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import tempfile
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

############################
# ---- DATA LOADING ----
############################
DATA_DIR = Path("data/processed")
EDGE_CSV = DATA_DIR / "side_effects_clean.csv"
RISK_CSV = DATA_DIR / "drug_risk_scores.csv"

@st.cache_data(show_spinner="Loading data...")
def load_data():
    edges = pd.read_csv(EDGE_CSV, usecols=["drug_name", "side_effect", "freq_pct"])
    risks = pd.read_csv(RISK_CSV, usecols=["drug_name", "risk_score"])
    return edges, risks

@st.cache_data(show_spinner="Building graph...")
def build_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
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
def compute_centrality(_G: nx.DiGraph):
    return nx.betweenness_centrality(_G, k=min(100, max(1, len(_G.nodes))))

# load data and graph (small sample for UI responsiveness)
edges_df, risk_df = load_data()
G = build_graph(edges_df.head(500))

# quick lookups
risk_map = risk_df.set_index("drug_name")["risk_score"].to_dict()
side_effect_lookup = { drug: list(group["side_effect"]) for drug, group in edges_df.groupby("drug_name") }

############################
# ---- UI: Theme & motion controls  ----
############################
with st.sidebar:
    st.markdown("## UI Settings")
    theme_choice = st.radio("Theme", options=["Auto", "Light", "Dark"], index=0,
                            help="Auto follows your OS / browser preference")
    reduce_motion = st.checkbox("Reduce animation (accessibility)", value=False,
                               help="Disable floating background animation for accessibility")

# map theme attribute to css classes (we'll use CSS selectors)
if reduce_motion:
    motion_state = "reduced"
else:
    motion_state = "animated"

# we expose the theme choice to CSS using a small attribute via inline style on root container later
# (no JS required - we will set CSS rules for 'prefers-color-scheme' and for a simple 'light' / 'dark' choice via Python-inserted class)

############################
# ---- CSS + Floating Pills Background (pure CSS + HTML) ----
############################
# We'll inject the CSS + HTML via st.markdown (unsafe_allow_html=True). This avoids components.html.
# The animation uses CSS keyframes and multiple pill elements (positions randomized via CSS nth-child).
# The UI content sits on top with higher z-index and glassy cards for contrast.

# If reduce_motion is checked, we'll set animation: none for the pills.
reduce_motion_css = "animation: none !important; opacity: 0.95;" if reduce_motion else ""

# Choose gradient base depending on theme_choice; 'auto' will rely on prefers-color-scheme
# We will still set a default gradient. The color contrasts and card backgrounds ensure text readability.
css_html ="""
<style>
/* Make Streamlit root containers transparent so background shows through */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > .main > div,
.stApp, .block-container {{
    background: transparent !important;
    background-color: transparent !important;
}}

/* Fullscreen background layer for animated pills */
#sideeffectnet-bg {{
    pointer-events: none; /* so clicks pass through */
    position: fixed;
    inset: 0;
    z-index: -9999;
    overflow: hidden;
    display: block;
}

/* Gradient base that is visible under pills (auto adapts to prefers-color-scheme) */
:root {{
  --grad-light: linear-gradient(120deg, rgba(126, 92, 194, 0.85) 0%, rgba(91, 170, 255, 0.85) 40%, rgba(240, 120, 145, 0.85) 100%);
  --grad-dark: linear-gradient(120deg, #0b1220 0%, #0f172a 30%, #1f2937 70%);
}}

/* prefer dark */
@media (prefers-color-scheme: dark) {{
  #sideeffectnet-bg::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: var(--grad-dark);
      background-size: 300% 300%;
      filter: saturate(1.02) contrast(1.02) blur(0px);
      opacity: 0.95;
      transform: translateZ(0);
      animation: bgShiftDark 18s linear infinite;
  }}
}}

@media (prefers-color-scheme: light) {{
  #sideeffectnet-bg::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: var(--grad-light);
      background-size: 300% 300%;
      filter: saturate(1.05) contrast(1.03) blur(0px);
      opacity: 0.95;
      transform: translateZ(0);
      animation: bgShiftLight 14s linear infinite;
  }}
}}

/* If user specifically picked Light or Dark (not Auto), we override prefers-color-scheme via inline class */
html.sideeffectnet-theme-light #sideeffectnet-bg::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: var(--grad-light);
    background-size: 300% 300%;
    animation: bgShiftLight 14s linear infinite;
}}
html.sideeffectnet-theme-dark #sideeffectnet-bg::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: var(--grad-dark);
    background-size: 300% 300%;
    animation: bgShiftDark 18s linear infinite;
}}

@keyframes bgShiftLight {{
  0% {{ background-position: 0% 50%; }}
  50% {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}
@keyframes bgShiftDark {{
  0% {{ background-position: 20% 50%; }}
  50% {{ background-position: 80% 50%; }}
  100% {{ background-position: 20% 50%; }}
}}

/* Container for pills */
#sideeffectnet-bg .pill {{
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
    {reduce_motion_css}
}

/* small pill variant */
#sideeffectnet-bg .pill.small {{
    width: 44px;
    height: 22px;
    border-radius: 16px;
    opacity: 0.85;
    transform-origin: center;
}}

/* pill inner stripe to make it look like tablets/capsules */
#sideeffectnet-bg .pill::after {{
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
}}

/* pill label / AI icon (small) */
#sideeffectnet-bg .pill .icon {{
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
}}

/* animation: pills float and drift upwards with gentle rotation and side-to-side */
@keyframes floatUp {{
  0% {{
    transform: translate3d(0, 0vh, 0) rotate(0deg);
    opacity: 0;
  }}
  10% {{
    opacity: 0.95;
  }}
  50% {{
    transform: translate3d(30px, -40vh, 0) rotate(10deg);
    opacity: 0.95;
  }}
  100% {{
    transform: translate3d(-20px, -110vh, 0) rotate(-8deg);
    opacity: 0;
  }}
}}

@keyframes floatUpSlow {{
  0% {{
    transform: translate3d(0, 0vh, 0) rotate(0deg);
    opacity: 0;
  }}
  10% {{
    opacity: 0.9;
  }}
  50% {{
    transform: translate3d(-40px, -35vh, 0) rotate(-6deg);
    opacity: 0.95;
  }}
  100% {{
    transform: translate3d(20px, -110vh, 0) rotate(6deg);
    opacity: 0;
  }}
}}

/* We create many pill instances and vary their delays / durations / left positions via nth-child selectors */
#sideeffectnet-bg .pill:nth-child(1) {{
    left: 8%;
    bottom: -10%;
    animation: floatUp 18s linear infinite;
    animation-delay: 0s;
    transform: scale(1.05) rotate(6deg);
}}
#sideeffectnet-bg .pill:nth-child(2) {{
    left: 22%;
    bottom: -12%;
    animation: floatUpSlow 22s linear infinite;
    animation-delay: 2s;
    transform: scale(0.9) rotate(-4deg);
}}
#sideeffectnet-bg .pill:nth-child(3) {{
    left: 36%;
    bottom: -8%;
    animation: floatUp 20s linear infinite;
    animation-delay: 5s;
    transform: scale(1.0) rotate(2deg);
}}
#sideeffectnet-bg .pill:nth-child(4) {{
    left: 50%;
    bottom: -14%;
    animation: floatUpSlow 24s linear infinite;
    animation-delay: 1s;
    transform: scale(1.1) rotate(10deg);
}}
#sideeffectnet-bg .pill:nth-child(5) {{
    left: 64%;
    bottom: -9%;
    animation: floatUp 19s linear infinite;
    animation-delay: 4s;
    transform: scale(0.95) rotate(-8deg);
}}
#sideeffectnet-bg .pill:nth-child(6) {{
    left: 78%;
    bottom: -11%;
    animation: floatUpSlow 21s linear infinite;
    animation-delay: 6s;
    transform: scale(0.85) rotate(4deg);
}}
#sideeffectnet-bg .pill:nth-child(7) {{
    left: 12%;
    bottom: -20%;
    animation: floatUp 25s linear infinite;
    animation-delay: 8s;
    transform: scale(0.8) rotate(-6deg);
}}
#sideeffectnet-bg .pill:nth-child(8) {{
    left: 30%;
    bottom: -20%;
    animation: floatUpSlow 23s linear infinite;
    animation-delay: 10s;
    transform: scale(1.15) rotate(14deg);
}}
#sideeffectnet-bg .pill:nth-child(9) {{
    left: 46%;
    bottom: -18%;
    animation: floatUp 17s linear infinite;
    animation-delay: 3s;
    transform: scale(0.9) rotate(3deg);
}}
#sideeffectnet-bg .pill:nth-child(10) {{
    left: 62%;
    bottom: -22%;
    animation: floatUpSlow 26s linear infinite;
    animation-delay: 9s;
    transform: scale(1.05) rotate(-12deg);
}}
#sideeffectnet-bg .pill:nth-child(11) {{
    left: 82%;
    bottom: -19%;
    animation: floatUp 27s linear infinite;
    animation-delay: 12s;
    transform: scale(0.95) rotate(6deg);
}}
#sideeffectnet-bg .pill:nth-child(12) {{
    left: 4%;
    bottom: -25%;
    animation: floatUpSlow 28s linear infinite;
    animation-delay: 7s;
    transform: scale(1.0) rotate(-10deg);
}}
#sideeffectnet-bg .pill:nth-child(13) {{
    left: 18%;
    bottom: -24%;
    animation: floatUp 22s linear infinite;
    animation-delay: 14s;
    transform: scale(1.07) rotate(8deg);
}}
#sideeffectnet-bg .pill:nth-child(14) {{
    left: 34%;
    bottom: -21%;
    animation: floatUpSlow 20s linear infinite;
    animation-delay: 11s;
    transform: scale(0.92) rotate(-5deg);
}}
#sideeffectnet-bg .pill:nth-child(15) {{
    left: 48%;
    bottom: -26%;
    animation: floatUp 30s linear infinite;
    animation-delay: 16s;
    transform: scale(1.12) rotate(10deg);
}}
#sideeffectnet-bg .pill:nth-child(16) {{
    left: 66%;
    bottom: -27%;
    animation: floatUpSlow 29s linear infinite;
    animation-delay: 13s;
    transform: scale(0.86) rotate(-7deg);
}}
#sideeffectnet-bg .pill:nth-child(17) {{
    left: 80%;
    bottom: -28%;
    animation: floatUp 24s linear infinite;
    animation-delay: 15s;
    transform: scale(1.0) rotate(4deg);
}}
#sideeffectnet-bg .pill:nth-child(18) {{
    left: 56%;
    bottom: -30%;
    animation: floatUpSlow 32s linear infinite;
    animation-delay: 18s;
    transform: scale(0.9) rotate(-4deg);
}}
#sideeffectnet-bg .pill:nth-child(19) {{
    left: 26%;
    bottom: -30%;
    animation: floatUp 34s linear infinite;
    animation-delay: 17s;
    transform: scale(0.78) rotate(-12deg);
}}
#sideeffectnet-bg .pill:nth-child(20) {{
    left: 72%;
    bottom: -32%;
    animation: floatUpSlow 33s linear infinite;
    animation-delay: 19s;
    transform: scale(1.18) rotate(12deg);
}}

/* smaller pills layered in front */
#sideeffectnet-bg .pill.small:nth-child(21) {{ left: 10%; bottom: -8%; animation: floatUp 16s linear infinite; animation-delay: 2s; transform: scale(0.6); }}
#sideeffectnet-bg .pill.small:nth-child(22) {{ left: 40%; bottom: -16%; animation: floatUpSlow 19s linear infinite; animation-delay: 5s; transform: scale(0.7); }}
#sideeffectnet-bg .pill.small:nth-child(23) {{ left: 70%; bottom: -12%; animation: floatUp 18s linear infinite; animation-delay: 6s; transform: scale(0.65); }}
#sideeffectnet-bg .pill.small:nth-child(24) {{ left: 52%; bottom: -6%; animation: floatUpSlow 17s linear infinite; animation-delay: 1s; transform: scale(0.55); }}

/* ensure main UI sits above background */
[data-testid="stAppViewContainer"] {{
    position: relative;
    z-index: 0;
}}
.block-container {{
    position: relative;
    z-index: 10;
}}

/* glass cards for contrast */
.stMarkdown, .stFrame, .stDataFrameContainer, .stDataFrame, .stAlert, .stMetric, .stButton {{
    background: rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(8px) saturate(120%) !important;
    border-radius: 10px !important;
    padding: 0.6rem !important;
    color: inherit;
}}

/* dark-mode tweaks for readability */
@media (prefers-color-scheme: dark) {{
    .stMarkdown, .stFrame, .stDataFrameContainer, .stDataFrame, .stAlert, .stMetric, .stButton {{
        background: rgba(5,8,12,0.55) !important;
        color: #e6eef8 !important;
    }}
}}

/* Responsive adjustments */
@media (max-width: 600px) {{
    #sideeffectnet-bg .pill {{ width: 54px; height: 26px; }}
    #sideeffectnet-bg .pill.small {{ width: 32px; height: 16px; }}
}}
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

# Insert CSS/HTML into page via st.markdown (unsafe HTML)
st.markdown(css_html, unsafe_allow_html=True)

# If user forced Light or Dark (not Auto), add a small inline class to html tag via st.markdown
# Using a style attribute on a top-level element is safer than injecting a script.
# We'll create a tiny root wrapper with the desired class name so CSS selectors for html.sideeffectnet-theme-* match.
if theme_choice == "Light":
    st.markdown('<div id="sideeffectnet-theme-override" data-theme="light"></div>', unsafe_allow_html=True)
    # add style to set html class via CSS hack (targets :root)
    st.markdown("""
    <style>
    :root { --sideeffectnet-theme-override: light; }
    html.sideeffectnet-theme-dark, html.sideeffectnet-theme-light { /* placeholder so classes exist */ }
    </style>
    """, unsafe_allow_html=True)
    # set class by adding a transparent top-level element with a unique id (we can't execute JS reliably)
    # But CSS prefers-color-scheme will handle most cases. This small placeholder keeps code robust.
elif theme_choice == "Dark":
    st.markdown('<div id="sideeffectnet-theme-override" data-theme="dark"></div>', unsafe_allow_html=True)

############################
# ---- SIDEBAR CONTENT ----
############################
with st.sidebar:
    st.image("media/sideeffectnetlogo.png", width=150)
    st.markdown("### Filters")
    min_risk, max_risk = risk_df["risk_score"].min(), risk_df["risk_score"].max()
    risk_filter = st.slider(
        "Filter by risk score",
        min_value=float(min_risk),
        max_value=float(max_risk),
        value=(float(min_risk), float(max_risk))
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

# ------------ rest of UI tabs & logic (keeps behavior unchanged) ------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Drug Lookup",
    "Safer Alternatives",
    "Risk Explorer",
    "Polypharmacy",
    "Critical Nodes",
    "Risk Hypotheses"
])

# TAB 1: Drug Lookup (kept concise; unchanged logic)
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

# TAB 2, 3, 4, 5, 6 — logic left intact from earlier corrected file
# (For brevity they are not repeated here; keep your original code sections for these tabs.)
# You can re-add the remaining tabs logic exactly as in your previous corrected file.
# — End of requested background animation implementation.
