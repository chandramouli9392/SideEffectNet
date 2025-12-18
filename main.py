import streamlit as st

from src.qa import load_qa
from src.graph_builder import build_side_effect_graph
from src.visualize_graph import visualize_graph, visualize_complete_graph
from src.analytics import risk_scores
from src.risk_analyzer import (
    calculate_and_add_risk_scores,
    export_risk_scores,
    visualize_risk_scores
)

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="SideEffectNet",
    layout="wide"
)

st.title("💊 SideEffectNet – Drug Safety Intelligence System")

# -------------------------------
# Cache heavy operations
# -------------------------------
@st.cache_resource
def load_graph():
    g = build_side_effect_graph("data/processed/side_effects_clean.csv")
    g = calculate_and_add_risk_scores(g)
    return g

@st.cache_resource
def load_rag():
    return load_qa()

graph = load_graph()
qa_bot = load_rag()

# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs([
    "Drug Lookup",
    "Risk Explorer",
    "Graphs",
    "Ask SideEffectNet 🤖"
])

# -------------------------------
# Tab 0 – Drug Lookup
# -------------------------------
with tabs[0]:
    st.subheader("🔍 Drug Lookup")
    st.write("Explore drugs and their connected side effects using the graph.")

    st.write(f"Total Nodes: {len(graph.nodes())}")
    st.write(f"Total Edges: {len(graph.edges())}")

# -------------------------------
# Tab 1 – Risk Explorer
# -------------------------------
with tabs[1]:
    st.subheader("⚠️ Risk Explorer")

    top_drugs = risk_scores(graph)[:10]
    for drug, score in top_drugs:
        st.write(f"**{drug}** — Risk Score: {score:.2f}")

# -------------------------------
# Tab 2 – Graphs
# -------------------------------
with tabs[2]:
    st.subheader("📊 Graph Visualizations")

    if st.button("Generate Risk Graphs"):
        export_risk_scores(graph, output_csv="drug_risk_scores.csv")
        visualize_graph(graph, output_path="sideeffectnet_graph.html", max_nodes=300)
        visualize_risk_scores(
            "drug_risk_scores.csv",
            output_html="risk_scores_graph.html"
        )
        visualize_complete_graph(
            graph,
            output_path="complete_sideeffectnet_graph.html"
        )
        st.success("Graphs generated successfully!")

# -------------------------------
# Tab 3 – RAG BOT (THIS IS WHAT YOU WANTED)
# -------------------------------
with tabs[3]:
    st.subheader("🤖 Ask SideEffectNet (RAG)")
    st.caption(
        "Ask questions about drug side effects. "
        "Answers are generated from verified medical documents."
    )

    question = st.text_input(
        "Ask about drug side effects",
        placeholder="e.g. What are the side effects of ibuprofen?"
    )

    if st.button("Ask"):
        if question.strip():
            with st.spinner("Searching medical documents..."):
                answer = qa_bot.run(question)
            st.success(answer)
