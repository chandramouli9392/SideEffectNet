# SideEffectNet

## Overview
SideEffectNet is an intelligent scientific assistant designed to analyze drug safety and visualize drug-side effect relationships. It provides tools for risk analysis, hypothesis generation, and data visualization, making it a powerful platform for pharmacological research and decision-making.

## Features
- **Drug Risk Analysis**: Analyze risk scores and side effects for individual drugs.
- **Drug Interaction Analysis**: Identify shared side effects between drug combinations.
- **Risk Visualization**: Generate bar charts and network graphs to visualize drug risk scores and relationships.
- **Polypharmacy Risk Detection**: Detect potential risks when combining multiple medications.
- **Hypothesis Generation**: Generate scientifically validated hypotheses for drug combinations.

## Project Structure
```
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
├── README.MD
├── requirements.txt
├── risk_analysis_report.pdf
├── risk_scores_graph.html
└── sideeffectnet_graph.html
```

## Usage
### Running the CLI
1. Run the main script:
   ```bash
   python src/plugin.py
   ```
2. Follow the prompts to select actions such as:
   - Analyze Risk
   - Generate Hypotheses
   - Generate PDF
   - Validate Data
   - Generate Graph

### Running the Dashboard
1. Start the Streamlit dashboard:
   ```bash
   streamlit run src/dashboard.py
   ```
2. Open the dashboard in your browser to explore drug-side effect relationships and risk scores.

## Dependencies
The project uses the following Python libraries:
- Streamlit
- Pandas
- NetworkX
- PyVis
- Plotly
- st-aggrid
- Google GenAI
- Dotenv
- ReportLab
- FPDF
- Matplotlib
- Colorama

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or support, please contact:
- **Chandramouli boppana**
- GitHub:https://github.com/chandramouli9392
- Email:chandramouliboppana7@gmail.com

## Acknowledgments
- FDA Adverse Event Reporting System for data sources.
- Contributors and developers who made this project possible.

## How the Project Was Built

### Data Processing
The project begins with raw datasets sourced from the FDA Adverse Event Reporting System and other sources. These datasets include:
- `drug_names.tsv`: Contains raw drug names.
- `meddra_all_se.tsv`: Contains all reported side effects.
- `meddra_freq.tsv`: Contains frequency data for side effects.

These raw datasets are processed using `data_processing.py` to clean and structure the data into usable formats:
- `drug_risk_scores.csv`: Contains risk scores for drugs on a scale of 0 to 1, where higher scores indicate greater risk.
- `side_effects_clean.csv`: Maps drugs to their associated side effects, cleaned for analysis.

These datasets are essential for constructing subgraphs, analyzing risks, and generating visualizations.

### Graph Construction
Instead of building a complete knowledge graph (KG) with all nodes and edges, the project focuses on subgraph-based analysis to improve performance and maintain accuracy. Subgraphs are dynamically constructed using `graph_builder.py` based on user input or specific queries.

### Risk Analysis
`risk_analyzer.py` computes risk scores and identifies shared side effects between drugs. This module is essential for understanding drug safety and potential interactions.

### Visualization
Interactive visualizations are created using `visualize_graph.py`. These include:
- Network graphs to explore drug-side effect relationships.
- Bar charts to visualize risk scores.

### Dashboard
The Streamlit-based dashboard (`dashboard.py`) provides an interactive UI for:
- Exploring drug-side effect relationships.
- Viewing risk scores.
- Detecting polypharmacy risks.

### Command-Line Interface (CLI)
`plugin.py` implements a CLI for performing various actions, including:
- Risk analysis.
- Hypothesis generation.
- Graph visualization.
- Data validation.

### Hypothesis Generation
The project uses AI-powered tools, such as Google GenAI, to generate scientifically validated hypotheses for drug combinations. This feature is implemented in `plugin.py` and integrated into the dashboard.

### Key Features
1. **Subgraph-Based Analysis**:
   - Focuses on targeted graph construction to improve performance and maintain accuracy.
2. **Streamlit Dashboard**:
   - Provides an interactive UI for exploring drug safety data.
3. **Command-Line Interface**:
   - Allows users to perform risk analysis and generate reports directly from the terminal.
4. **Hypothesis Generation**:
   - Uses AI-powered tools to generate scientific hypotheses for drug combinations.

### Knowledge Graph Statistics
The knowledge graph constructed by SideEffectNet contains:
- **Total Nodes**: 7470
- **Total Edges**: 158559

These statistics highlight the scale of the graph and the depth of relationships analyzed within the project.

## Data Sources

The raw datasets used in this project were sourced from:

1. **SIDER Dataset**:
   - A comprehensive dataset for drug-side effect relationships.
   - [Learn more about SIDER](https://paperswithcode.com/dataset/sider).

2. **Side Effects Database**:
   - TSV files (`drug_names.tsv`, `meddra_all_se.tsv`, `meddra_freq.tsv`) were downloaded from [http://sideeffects.embl.de/download/](http://sideeffects.embl.de/download/).

