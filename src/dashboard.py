############################################
# dashboard.py (FINAL — Full working app with SIMPLE floating pills background)
############################################

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
        "Get Help": "https://github.com/ganeshmysoreDT/SideEffectNet",
        "Report a bug": "https://github.com/ganeshmysoreDT/SideEffectNet/issues",
        "About": "# SideEffectNet: Drug Safety Analytics Platform",
    },
)

# -----------------------
# Environment / API key
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
# Data loading utilities
# -----------------------
@st.cache_data(show_spinner="Loading data...")
def load_data():
    if not EDGE_CSV.exists() or not RISK_CSV.exists():
        return pd.DataFrame(), pd.DataFrame()

    edges = pd.read_csv(EDGE_CSV, usecols=["drug_name", "side_effect", "freq_pct"])
    risks = pd.read_csv(RISK_CSV, usecols=["drug_name", "risk_score"])
    return edges, risks


@st.cache_data(show_spinner="Building graph...")
def build_graph(edges_df):
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        drug = str(row["drug_name"])
        se = str(row["side_effect"])
        freq = row.get("freq_pct", "N/A")

        try:
            freq = float(freq)
        except:
            freq = "N/A"

        G.add_node(drug, type="drug")
        G.add_node(se, type="side_effect")
        G.add_edge(drug, se, frequency=freq)

    return G


@st.cache_data(show_spinner="Computing centrality...")
def compute_centrality(G):
    try:
        return nx.betweenness_centrality(G, k=min(100, len(G.nodes)))
    except:
        return nx.betweenness_centrality(G)


# -----------------------
# Load the data
# -----------------------
edges_df, risk_df = load_data()
G = build_graph(edges_df.head(500))

risk_map = risk_df.set_index("drug_name")["risk_score"].to_dict()
side_effect_lookup = {
    d: list(g["side_effect"]) for d, g in edges_df.groupby("drug_name")
}

# -----------------------
# 🌟 SIMPLE FLOATING PILLS BACKGROUND
# -----------------------
st.markdown(
    """
<style>
#pill-bg-container {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    overflow: hidden;
    z-index: -9999;
    pointer-events: none;
}

/* Single pill */
.pill {
    position: absolute;
    width: 60px; height: 24px;
    border-radius: 20px;
    background: linear-gradient(90deg, #ffffffcc, #e2e2ffcc);
    box-shadow: 0 4px 12px #0004;
    animation: floatUp 10s linear infinite;
}

/* Animation */
@keyframes floatUp {
    0%   { transform: translateY(110vh) rotate(0deg); opacity: 0; }
    10%  { opacity: 1; }
    50%  { transform: translateY(40vh) rotate(10deg); opacity: 1; }
    100% { transform: translateY(-20vh) rotate(-10deg); opacity: 0; }
}

/* Smaller pills for depth */
@media (max-width: 600px) {
    .pill { width: 40px; height: 16px; }
}
</style>

<div id="pill-bg-container">
    <!-- 20 floating pills -->
    <div class="pill" style="left:5%; animation-delay:0s;"></div>
    <div class="pill" style="left:15%; animation-delay:1s;"></div>
    <div class="pill" style="left:25%; animation-delay:2s;"></div>
    <div class="pill" style="left:35%; animation-delay:3s;"></div>
    <div class="pill" style="left:45%; animation-delay:4s;"></div>
    <div class="pill" style="left:55%; animation-delay:5s;"></div>
    <div class="pill" style="left:65%; animation-delay:6s;"></div>
    <div class="pill" style="left:75%; animation-delay:7s;"></div>
    <div class="pill" style="left:85%; animation-delay:8s;"></div>
    <div class="pill" style="left:90%; animation-delay:9s;"></div>

    <!-- Smaller pills -->
    <div class="pill" style="left:10%; animation-delay:3s; transform:scale(.7);"></div>
    <div class="pill" style="left:20%; animation-delay:6s; transform:scale(.8);"></div>
    <div class="pill" style="left:30%; animation-delay:9s; transform:scale(.6);"></div>
    <div class="pill" style="left:50%; animation-delay:4s; transform:scale(.9);"></div>
    <div class="pill" style="left:70%; animation-delay:2s; transform:scale(.8);"></div>
    <div class="pill" style="left:85%; animation-delay:5s; transform:scale(.7);"></div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:
    try:
        st.image("media/sideeffectnetlogo.png", width=150)
    except:
        st.write("SideEffectNet")

    st.markdown("### Filter by Risk")
    if not risk_df.empty:
        rmin, rmax = risk_df["risk_score"].min(), risk_df["risk_score"].max()
        risk_filter = st.slider("Risk Score Range", float(rmin), float(rmax), (float(rmin), float(rmax)))
    else:
        risk_filter = (0.0, 1.0)

    st.markdown("---")
    st.markdown("SideEffectNet analyzes:")
    st.markdown("- Drug–Side effect relationships")
    st.markdown("- Polypharmacy interactions")
    st.markdown("- Risk scoring")

# -----------------------
# MAIN PAGE HEADER
# -----------------------
st.title("SideEffectNet: Drug Safety Analytics")
st.markdown("Explore drug interactions, risks, and polypharmacy patterns.")


########################################################
# TAB 1 — DRUG LOOKUP
########################################################
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Drug Lookup",
        "Safer Alternatives",
        "Risk Explorer",
        "Polypharmacy",
        "Critical Nodes",
        "Risk Hypotheses",
    ]
)

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        drug_list = sorted(risk_map.keys())
        drug = st.selectbox("Select Drug", [""] + drug_list)

        if drug:
            score = risk_map.get(drug, 0)
            st.subheader(f"{drug} — Risk Score: {score:.3f}")
            st.write("### Side Effects:")
            for s in side_effect_lookup.get(drug, [])[:20]:
                st.write("-", s)

    with col2:
        if drug:
            sg = nx.Graph()
            sg.add_node(drug)
            for se in side_effect_lookup.get(drug, [])[:20]:
                sg.add_node(se)
                sg.add_edge(drug, se)

            pv = Network(height="600px", width="100%", directed=False)
            pv.from_nx(sg)
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
                pv.save_graph(f.name)
                components.html(open(f.name).read(), height=600)


########################################################
# TAB 2 — SAFER ALTERNATIVES
########################################################
with tab2:
    st.header("Safer Alternatives")
    drug = st.selectbox("Select Drug", [""] + list(risk_map.keys()))

    if drug:
        target_side_effects = set(side_effect_lookup.get(drug, []))
        risk_original = risk_map[drug]

        alt_list = []
        for d, effects in side_effect_lookup.items():
            if d == drug:
                continue
            overlap = len(target_side_effects & set(effects))
            if overlap > 0 and risk_map[d] < risk_original:
                alt_list.append([d, overlap, risk_map[d]])

        if alt_list:
            df = pd.DataFrame(alt_list, columns=["Drug", "Shared Effects", "Risk"])
            df = df.sort_values(["Shared Effects", "Risk"], ascending=[False, True])
            st.dataframe(df)
        else:
            st.info("No safer alternatives found.")


########################################################
# TAB 3 — RISK EXPLORER
########################################################
with tab3:
    st.header("Risk Explorer")
    filt_df = risk_df[
        (risk_df["risk_score"] >= risk_filter[0])
        & (risk_df["risk_score"] <= risk_filter[1])
    ]
    st.dataframe(filt_df)

    fig = px.histogram(filt_df, x="risk_score", nbins=30)
    st.plotly_chart(fig)


########################################################
# TAB 4 — POLYPHARMACY
########################################################
with tab4:
    st.header("Polypharmacy Interaction Checker")
    drugs = st.multiselect("Select drugs", list(risk_map.keys()))

    if len(drugs) >= 2:
        combined = set()
        overlap = None
        total_risk = 0

        for d in drugs:
            effects = set(side_effect_lookup.get(d, []))
            combined |= effects
            overlap = effects if overlap is None else overlap & effects
            total_risk += risk_map[d]

        st.write("### Total Side Effects:", len(combined))
        st.write("### Shared Side Effects:", len(overlap))
        st.write("### Combined Risk Score:", round(total_risk, 3))


########################################################
# TAB 5 — CRITICAL NODES
########################################################
with tab5:
    st.header("Critical Nodes (Centrality)")
    cent = compute_centrality(G)
    df = pd.DataFrame(
        [{"node": n, "centrality": c, "type": G.nodes[n]["type"]} for n, c in cent.items()]
    )

    st.write("### Top Drugs")
    st.dataframe(df[df["type"] == "drug"].nlargest(10, "centrality"))

    st.write("### Top Side Effects")
    st.dataframe(df[df["type"] == "side_effect"].nlargest(10, "centrality"))


########################################################
# TAB 6 — AI HYPOTHESES
########################################################
with tab6:
    st.header("AI-Generated Drug Interaction Hypotheses")

    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Primary Drug", [""] + list(risk_map.keys()))
    with c2:
        b = st.selectbox("Combination Drug", [""] + list(risk_map.keys()))

    if a and b:
        overlap = set(side_effect_lookup[a]) & set(side_effect_lookup[b])

        st.subheader("Shared Effects:")
        st.write(len(overlap))

        if st.button("Generate AI Hypotheses"):
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                prompt = f"""
Analyze interaction between:

Drug A: {a} (Risk {risk_map[a]:.2f})
Drug B: {b} (Risk {risk_map[b]:.2f})

Shared Effects: {list(overlap)}

Generate 3 scientific hypotheses explaining:
- Mechanisms
- Pharmacodynamic interaction
- CYP450 issues
- Clinical significance
"""

                res = client.models.generate_content(
                    model="gemini-2.0-flash", contents=prompt
                )

                st.markdown(res.text)

            except Exception as e:
                st.error(f"Gemini API Error: {e}")
