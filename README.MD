🌐 SideEffectNet
🔬 AI-Powered Drug Safety Intelligence Platform

SideEffectNet is an intelligent scientific assistant designed to analyze drug safety, detect polypharmacy risks, and visualize drug–side effect networks with interactive graphs.
It supports researchers, clinicians, and pharmacologists in decision-making, hypothesis generation, and risk prediction.

🚀 Overview

SideEffectNet provides:

📊 Drug Risk Analysis

🔗 Drug–Drug Interaction Detection

🌐 Side Effect Network Visualization

🧠 AI-Generated Scientific Hypotheses

⚠️ Polypharmacy Risk Detection

📄 Automated PDF Risk Reports

Built on real-world FDA adverse event datasets, it brings explainable safety intelligence into a user-friendly platform.

⭐ Features
🔍 1. Drug Risk Analysis

Analyze risk scores and associated side effects for individual drugs.

🔗 2. Drug Interaction Analysis

Identify shared side effects between drug pairs or combinations.

📈 3. Risk Visualization

Generate:

Bar charts

Polypharmacy risk graphs

Interactive network maps

⚠️ 4. Polypharmacy Risk Detection

Automatic screening for risk when multiple medications are combined.

🧪 5. Hypothesis Generation (AI-Powered)

Using Google GenAI, the system suggests:

Mechanistic hypotheses

Shared pathway explanations

Possible ADR causes

🧰 6. Command-Line Interface (CLI)

Perform:

Risk analysis

Graph generation

Hypothesis generation

Data validation

PDF report generation

🖥️ 7. Streamlit Dashboard

A clean, interactive interface for live network exploration & risk scoring.

🗂️ Project Structure
sideeffectnet/
├── data/
│   ├── processed/
│   │   ├── drug_risk_scores.csv
│   │   └── side_effects_clean.csv
│   ├── raw/
│   │   ├── drug_names.tsv
│   │   ├── meddra_all_se.tsv
│   │   └── meddra_freq.tsv
├── media/
│   └── sideeffectnetlogo.png
├── src/
│   ├── analytics.py
│   ├── dashboard.py
│   ├── data_processing.py
│   ├── graph_builder.py
│   ├── plugin.py
│   ├── risk_analyzer.py
│   ├── utils.py
│   ├── visualize_graph.py
│   └── __pycache__/
├── drug_risk_scores.csv
├── main.py
├── README.md
├── requirements.txt
├── risk_analysis_report.pdf
├── risk_scores_graph.html
└── sideeffectnet_graph.html

🧑‍💻 Usage
▶️ Running the CLI
python src/plugin.py


Then choose actions:

1️⃣ Analyze Risk

2️⃣ Generate Hypotheses

3️⃣ Generate PDF Report

4️⃣ Validate Dataset

5️⃣ Generate Graph

🖥️ Running the Dashboard
streamlit run src/dashboard.py


Then open the local URL to explore:

Drug risk scores

Side-effect networks

Polypharmacy red-flags

📦 Dependencies

SideEffectNet uses:

Library	Purpose
Streamlit	Dashboard
Pandas	Data processing
NetworkX	Graph construction
PyVis	Visual network rendering
Plotly	Interactive graphs
st-aggrid	Data table display
Google GenAI	Hypothesis generation
Dotenv	Env variable handling
ReportLab / FPDF	PDF generation
Matplotlib	Visualization
Colorama	CLI styling

Install them using:

pip install -r requirements.txt

🛠️ How SideEffectNet Was Built
📥 1. Data Processing

Raw datasets from FDA & SIDER:

drug_names.tsv – drug list

meddra_all_se.tsv – all side effects

meddra_freq.tsv – frequency mappings

Cleaned into:

drug_risk_scores.csv

side_effects_clean.csv

Using data_processing.py.

🧩 2. Graph Construction

Instead of building a massive full KG (slow & heavy), SideEffectNet uses dynamic subgraphs for:

Drug-drug queries

Risk-based filtering

Edge-relevance trimming

Done via graph_builder.py.

🧮 3. Risk Analysis

risk_analyzer.py calculates:

Single drug risk score (0–1)

Interaction risk

Shared ADRs

Polypharmacy indicators

🌐 4. Visualization

visualize_graph.py generates:

Network graphs (PyVis)

Bar charts (Plotly)

Summary graphs (HTML export)

🧬 5. Hypothesis Generation

Using AI-powered GenAI, SideEffectNet can propose:

Possible mechanistic overlaps

Shared binding targets

Pharmacokinetic explanations

Integrated via plugin.py.

📊 Knowledge Graph Stats
Metric	Value
Total Nodes	7,470
Total Edges	158,559

This reflects the depth and richness of drug–side effect relationships analyzed.

📚 Data Sources

SIDER Dataset
Comprehensive drug-side effect resource.
🔗 http://sideeffects.embl.de/download/

FDA Adverse Event Reporting System (FAERS)
Source of raw ADR data.

🤝 Contributing

We welcome contributions!

🍴 Fork the repository

🌱 Create a new feature branch

🛠️ Make changes

📩 Submit a PR with detailed notes

📜 License

This project is licensed under the MIT License.

📬 Contact

👤 Chandramouli Boppana
🔗 GitHub: https://github.com/chandramouli9392

📧 Email: chandramouliboppana7@gmail.com

🙏 Acknowledgments

FDA Adverse Event Reporting System

SIDER Dataset creators

All contributors who made this project possible
