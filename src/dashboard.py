import streamlit as st
st.set_page_config(
    page_title="SideEffectNet Dashboard", 
    layout="wide",
    page_icon="media/sideeffectnetlogo.png",  # Set the logo as the tab icon
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
# ---- DATA LOADING ----   #
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
        drug = row["drug_name"].strip()
        se = row["side_effect"].strip()
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
    return nx.betweenness_centrality(_G, k=min(100, len(_G.nodes)))

edges_df, risk_df = load_data()
G = build_graph(edges_df.head(500))  # Reduced size for performance

# Precompute lookup dictionaries
risk_map = risk_df.set_index("drug_name")["risk_score"].to_dict()
side_effect_lookup = {
    drug: list(group["side_effect"])
    for drug, group in edges_df.groupby("drug_name")
}

############################
# ---- STREAMLIT UI ----   #
############################

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        border: 1px solid #e1e4e8;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #f6f8fa;
    }
    .stMetric label {
        font-size: 1rem !important;
        color: #57606a !important;
    }
    .stMetric div {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .css-1v0mbdj {
        border-radius: 0.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .st-b7 {
        color: #24292f;
    }
    .st-b8 {
        color: #57606a;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    .ag-body.ag-layout-normal {
        width: 100% !important;
        height: 100vh !important;
        margin: 0;
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with filters and info
with st.sidebar:
    st.image("media/sideeffectnetlogo.png", width=150)  # Display the logo from the media folder
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

# Main content
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

############################################
#  TAB 1: Drug Search + Explanation + Subgraph
############################################
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Drug Profile")
        drug_list = sorted(set(risk_map.keys()))
        
        drug = st.selectbox(
            "Search for a drug:",
            options=[""] + drug_list,
            index=0,
            key="tab1_drug_search"
        )
        
        if drug:
            if drug not in risk_map:
                st.error("Drug not found in dataset.")
            else:
                score = risk_map[drug]
                
                # Risk score with color coding
                risk_color = "red" if score > 0.7 else "orange" if score > 0.4 else "green"
                st.markdown(f"""
                <div style="border-radius: 0.5rem; padding: 1rem; background-color: #f8f9fa; border-left: 0.3rem solid {risk_color}">
                    <h3 style="margin-top: 0; color: #24292f;">{drug}</h3>
                    <div style="font-size: 2rem; font-weight: bold; color: {risk_color};">{score:.3f}</div>
                    <div style="color: #57606a;">Risk Score (0-1 scale)</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Side effects list
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
                sg.add_node(drug, color="#636EFA", size=30, title=f"Drug: {drug}\nRisk: {risk_map.get(drug, 'N/A')}")
                
                for se in se_list[:20]:  # Limit to 20 for performance
                    sg.add_node(se, color="#EF553B", size=20, title=f"Side Effect: {se}")
                    if G.has_edge(drug, se):
                        freq = G.edges[drug, se].get("frequency", "N/A")
                        sg.add_edge(drug, se, value=freq if isinstance(freq, (int, float)) else 1, 
                                  title=f"Frequency: {freq}%")
                    else:
                        sg.add_edge(drug, se, value=1, title="Frequency: N/A")

                # Improved PyVis network
                pv = Network(
                    height="600px", 
                    width="100%", 
                    bgcolor="white", 
                    font_color="black",
                    directed=False,
                    notebook=False
                )
                pv.from_nx(sg)
                
                # Enhanced physics configuration
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
                
                # Save and display
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
                    pv.save_graph(tmpfile.name)
                    components.html(open(tmpfile.name, 'r').read(), height=620, scrolling=False)
            else:
                st.warning("No side effects to visualize for this drug.")

############################################
#  TAB 2: Suggest Safer Alternatives
############################################
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
                sugg_df = pd.DataFrame(suggestions).sort_values(
                    ["Shared Effects", "Risk Reduction"], 
                    ascending=[False, False]
                )

                st.dataframe(sugg_df, use_container_width=True)
            else:
                st.info("No safer alternatives with overlapping side effects found.")
    else:
        st.info("Please select a drug in the 'Drug Lookup' tab first.")

############################################
#  TAB 3: Risk Score Filter Explorer
############################################
# Fix filtering logic and update table configuration
with tab3:
    st.header("Drug Risk Explorer")
    st.markdown("Analyze and compare drug risk scores across the dataset.")

    # Correct filtering logic
    filtered = risk_df[(risk_df["risk_score"] >= risk_filter[0]) & 
                      (risk_df["risk_score"] <= risk_filter[1])]

    col1, col2 = st.columns([1, 3])
    with col1:
        # Determine color for "Drugs in Range" based on count
        drugs_in_range_color = "green" if len(filtered) > 50 else "orange" if len(filtered) > 20 else "red"
        # Styled container for "Drugs in Range"
        st.markdown(f"""
        <div style="border-radius: 0.5rem; padding: 1rem; background-color: #ffffff; border-left: 0.3rem solid {drugs_in_range_color}; margin-bottom: 1rem;">
            <div style="font-size: 1rem; color: #57606a;">Drugs in Range</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {drugs_in_range_color};">{len(filtered)}</div>
        </div>
        """, unsafe_allow_html=True)

        # Determine color for "Average Risk" based on risk level
        avg_risk = filtered["risk_score"].mean()
        avg_risk_color = "green" if avg_risk <= 0.4 else "orange" if avg_risk <= 0.7 else "red"
        # Styled container for "Average Risk"
        st.markdown(f"""
        <div style="border-radius: 0.5rem; padding: 1rem; background-color: #ffffff; border-left: 0.3rem solid {avg_risk_color}; margin-bottom: 1rem;">
            <div style="font-size: 1rem; color: #57606a;">Average Risk</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {avg_risk_color};">{avg_risk:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Top 5 highest risk
        st.markdown("**Highest Risk in Range**")
        for _, row in filtered.nlargest(5, "risk_score").iterrows():
            st.markdown(f"- {row['drug_name']} ({row['risk_score']:.3f})")

    with col2:
        # Interactive histogram
        fig = px.histogram(
            filtered, 
            x="risk_score",
            nbins=30,
            title="Distribution of Risk Scores",
            labels={"risk_score": "Risk Score"},
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(
            bargap=0.1,
            yaxis_title="Number of Drugs",
            xaxis_range=[risk_filter[0]-0.05, risk_filter[1]+0.05]
        )
        st.plotly_chart(fig, use_container_width=True)

    # Consistent table theme
    st.markdown("### Drug Risk Data")
    st.dataframe(filtered, use_container_width=True)

############################################
#  TAB 4: Polypharmacy Risk Detection
############################################
with tab4:
    st.header("Polypharmacy Risk Analyzer")
    st.markdown("Identify potential risks when combining multiple medications.")
    
    # Add color legend
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
    
    drug_options = sorted(risk_df["drug_name"].unique())
    selected_drugs = st.multiselect(
        "Select 2-5 drugs to analyze combinations", 
        drug_options, 
        max_selections=5,
        help="Select multiple drugs to check for overlapping side effects"
    )
    
    if len(selected_drugs) >= 2:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Calculate combined metrics
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
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: #ffffff; border-left: 0.3rem solid blue; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Average Risk</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: blue;">{avg_score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: #ffffff; border-left: 0.3rem solid red; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Highest Individual Risk</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: red;">{max_score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: #ffffff; border-left: 0.3rem solid green; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Total Unique Side Effects</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: green;">{len(combined_effects)}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: #ffffff; border-left: 0.3rem solid orange; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Overlapping Side Effects</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: orange;">{len(overlap_effects) if overlap_effects else 0}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Risk indicator
            risk_level = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.4 else "Low"
            color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            
            st.markdown(f"""
            <div style="border-radius: 0.5rem; padding: 1rem; background-color: #ffffff; border-left: 0.3rem solid {color}; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #57606a;">Combined Risk Level</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if len(selected_drugs) > 2:
                st.warning("Combining more than 2 drugs increases risk exponentially")
        
        # Visualizations
        tab1, tab2 = st.tabs(["Side Effect Overlap", "Risk Comparison"])
        
        with tab1:
            if overlap_effects:
                st.markdown("### Overlapping Side Effects")
                for i, effect in enumerate(list(overlap_effects)[:20], start=1):
                    st.markdown(f"- {effect}")
                if len(overlap_effects) > 20:
                    st.markdown(f"... and {len(overlap_effects)-20} more")
            else:
                st.success("No overlapping side effects detected among selected drugs.")
        
        with tab2:
            # Radar chart for risk comparison
            fig = go.Figure()
            
            for d in selected_drugs:
                fig.add_trace(go.Scatterpolar(
                    r=[risk_map[d]],
                    theta=[d],
                    fill='toself',
                    name=d,
                    textfont=dict(color='black')  # Set text color to black
                ))
            
            # Update the horizontal axis numbers to make them black
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(color='black')  # Set tick font color to black
                    )
                ),
                showlegend=True,
                title="Individual Drug Risk Scores"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least 2 drugs to analyze combinations")

############################################
#  TAB 5: Highlight Critical Nodes (Centrality)
############################################
with tab5:
    st.header("Network Critical Nodes Analysis")
    st.markdown("Identify the most influential drugs and side effects in the network.")
    
    if "centrality" not in st.session_state:
        with st.spinner("Computing network centrality..."):
            st.session_state["centrality"] = compute_centrality(G)
    
    centrality = st.session_state["centrality"]
    cent_df = pd.DataFrame([
        {"Node": n, "Type": G.nodes[n]["type"], "Centrality": c}
        for n, c in centrality.items()
    ])
    
    # Top drugs and side effects
    drug_top = cent_df[cent_df["Type"] == "drug"].nlargest(10, "Centrality")
    se_top = cent_df[cent_df["Type"] == "side_effect"].nlargest(10, "Centrality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Most Central Drugs")
        st.markdown("Drugs that connect to many side effects in the network")
        
        fig = px.bar(
            drug_top,
            x="Centrality",
            y="Node",
            orientation='h',
            color="Centrality",
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Clinical Implications**")
        st.markdown("""
        - High centrality drugs may have broad side effect profiles
        - Potential for more drug-drug interactions
        - May require closer monitoring in clinical practice
        """)
    
    with col2:
        st.markdown("### Most Central Side Effects")
        st.markdown("Side effects associated with many different drugs")
        
        fig = px.bar(
            se_top,
            x="Centrality",
            y="Node",
            orientation='h',
            color="Centrality",
            color_continuous_scale='Plasma'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Clinical Implications**")
        st.markdown("""
        - Common side effects across drug classes
        - May represent general physiological responses
        - Potential targets for preventative therapies
        """)
    
    # Network visualization of top nodes
    if st.checkbox("Show Critical Nodes Network", key="critical_nodes_network_checkbox"):
        top_nodes = list(drug_top["Node"]) + list(se_top["Node"])
        subgraph = G.subgraph(top_nodes)
        
        # Advanced PyVis network visualization with restricted zoom-out
        pv = Network(
            height="700px", 
            width="100%", 
            bgcolor="white", 
            font_color="black",
            directed=False,
            notebook=False
        )

        # Add nodes and edges to the PyVis graph
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node]["type"]
            pv.add_node(
                node, 
                color="#636EFA" if node_type == "drug" else "#EF553B",
                size=30 if node_type == "drug" else 20,
                title=f"{node_type.capitalize()}: {node}\nCentrality: {centrality[node]:.4f}",
                shape="dot"
            )

        for edge in subgraph.edges():
            pv.add_edge(
                edge[0], 
                edge[1], 
                color="#A3A3A3", 
                width=2, 
                title=f"Edge between {edge[0]} and {edge[1]}"
            )

        # Save and display the graph
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
            pv.save_graph(tmpfile.name)
            components.html(open(tmpfile.name, 'r').read(), height=720, scrolling=False)


############################################
#  TAB 6: Risk Hypotheses (Professional Analysis)
############################################
with tab6:
    st.header("AI-Powered Risk Hypotheses")
    st.markdown("Generate scientifically validated hypotheses about drug combination risks.")

    # Add color legend
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
    
    # Ensure the active tab state is properly initialized and maintained
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = 'Risk Hypotheses'

    # Update tab selection logic
    if st.session_state['active_tab'] == 'Risk Hypotheses':
        with tab6:
            # Explicitly set the active tab to Risk Hypotheses
            st.session_state['active_tab'] = 'Risk Hypotheses'
            # Section 1: Drug Selection
            st.subheader("1. Select Drug Combination")
            col1, col2 = st.columns(2)
            with col1:
                drug_a = st.selectbox(
                    "Primary Drug",
                    options=[""] + sorted(risk_map.keys()),
                    index=0,
                    key="tab6_primary_drug"
                )
            with col2:
                drug_b = st.selectbox(
                    "Combination Drug", 
                    options=[""] + sorted(risk_map.keys()),
                    index=0,
                    key="tab6_secondary_drug"
                )
        
            if drug_a and drug_b:
                # Section 2: Safety Analysis
                st.subheader("2. Safety Analysis")
                
                # Calculate metrics
                risk_a = risk_map.get(drug_a, 0)
                risk_b = risk_map.get(drug_b, 0)
                side_effects_a = set(side_effect_lookup.get(drug_a, []))
                side_effects_b = set(side_effect_lookup.get(drug_b, []))
                overlapping = side_effects_a & side_effects_b
                
                # Display metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    risk_a_color = "red" if risk_a > 0.7 else "orange" if risk_a > 0.4 else "green"
                    st.markdown(f"""
                    <div style="border-radius: 0.5rem; padding: 1rem; background-color: #ffffff; border-left: 0.3rem solid {risk_a_color}; margin-bottom: 1rem;">
                        <div style="font-size: 1rem; color: #57606a;">{drug_a} Risk Score</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {risk_a_color};">{risk_a:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with metrics_col2:
                    risk_b_color = "red" if risk_b > 0.7 else "orange" if risk_b > 0.4 else "green"
                    st.markdown(f"""
                    <div style="border-radius: 0.5rem; padding: 1rem; background-color: #ffffff; border-left: 0.3rem solid {risk_b_color}; margin-bottom: 1rem;">
                        <div style="font-size: 1rem; color: #57606a;">{drug_b} Risk Score</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {risk_b_color};">{risk_b:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with metrics_col3:
                    overlap_color = "red" if len(overlapping) > 10 else "orange" if len(overlapping) > 5 else "green"
                    st.markdown(f"""
                    <div style="border-radius: 0.5rem; padding: 1rem; background-color: #ffffff; border-left: 0.3rem solid {overlap_color}; margin-bottom: 1rem;">
                        <div style="font-size: 1rem; color: #57606a;">Shared Side Effects</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {overlap_color};">{len(overlapping)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Section 3: Hypothesis Generation
                st.subheader("3. AI-Generated Hypotheses")
                
                # Remove the generate PDF button and improve the Gemini API response formatting
                if st.button("Generate Scientific Hypotheses", type="primary"):
                    with st.spinner("Analyzing pharmacological profiles..."):
                        # Prepare scientific context
                        context = {
                            "drug_a": drug_a,
                            "drug_b": drug_b,
                            "risk_a": risk_a,
                            "risk_b": risk_b,
                            "overlap_count": len(overlapping),
                            "overlap_effects": list(overlapping)[:10]  # Show top 10
                        }

                        # Professional prompt template
                        prompt_template = """
                        As a senior pharmacologist, analyze this drug combination:

                        **Drugs**: {drug_a} (Risk: {risk_a:.2f}) + {drug_b} (Risk: {risk_b:.2f})
                        **Shared Side Effects**: {overlap_count}
                        **Key Overlaps**: {overlap_effects}

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

                        client = genai.Client(api_key=GEMINI_API_KEY)

                        # Generate with Google GenAI
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=prompt_template.format(**context)
                        )

                        # Display formatted results
                        st.markdown("### Generated Hypotheses")
                        formatted_response = response.text.replace(
                            "Okay, let's analyze the drug combination of carnitine and anidulafungin and generate clinically-relevant hypotheses based on the provided information.",
                            ""
                        )
                        st.markdown(formatted_response)
            else:
                # Fix the logic for displaying the "Please select two drugs to analyze" message
                st.warning("Please select two drugs to analyze")
