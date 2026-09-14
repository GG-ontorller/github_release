# Attractor Landscape Recovery in Boolean Networks

This repository contains the simulation code and numerical results for the
study of sparse attractor basin recovery under XOR feedback intervention in
Boolean networks.

## Contents

- \`code/\`: Python scripts for random Boolean network generation and basin
  recovery analysis.
- \`data/modular_multistable_N50/\`: raw results, summary statistics, and figures
  for ensembles of 50 accepted modular multistable random Boolean networks.
- \`data/modular_multistable_5modules_fig/\`: data and figures for the
  representative five-module experiment.

The CSV files contain the raw network-level results and ensemble summaries.
The PDF files contain the recovery curves, synergy heatmaps, and ensemble
statistics reported in the manuscript. Pickle files store the representative
network objects used to generate the corresponding figures.

## Requirements

Python 3.9 or later with:

\`\`\`text
numpy
pandas
matplotlib
openpyxl
\`\`\`

## Reproduction

From the repository root, run:

\`\`\`bash
python code/modular_multistable_recovery.py --groups three --networks-per-group 50
\`\`\`

The scripts use a fixed random seed by default. Command-line options allow the
network size, module structure, acceptance criteria, and output directory to be
changed.
