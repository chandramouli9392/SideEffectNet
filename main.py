from src.qa import load_qa
import streamlit as st
from src.graph_builder import build_side_effect_graph
from src.visualize_graph import visualize_graph, visualize_complete_graph
from src.analytics import risk_scores  
from src.risk_analyzer import calculate_and_add_risk_scores, export_risk_scores, visualize_risk_scores

    # 1. Build the graph
    graph = build_side_effect_graph("data/processed/side_effects_clean.csv")

    print("Total nodes:", len(graph.nodes()))
    print("Total edges:", len(graph.edges()))

    for u, v, d in list(graph.edges(data=True))[:5]:
        print(f"{u} --({d['relation']})--> {v}")

    # 2. Add risk scores to drug nodes
    graph = calculate_and_add_risk_scores(graph)

    # 3. Export scores as CSV
    export_risk_scores(graph, output_csv="drug_risk_scores.csv")

    # 4. Visualize with updated risk-based sizing (optional)
    visualize_graph(graph, output_path="sideeffectnet_graph.html", max_nodes=300)
    visualize_risk_scores("drug_risk_scores.csv", output_html="risk_scores_graph.html")
    # 5. Visualize the complete graph (optional)
    visualize_complete_graph(graph, output_path="complete_sideeffectnet_graph.html")

    print("\nTop Drugs by Risk Score (Weighted by freq_pct):")
    for drug, score in risk_scores(graph)[:10]:
        print(f"{drug} — Risk Score: {score:.2f}")

        
