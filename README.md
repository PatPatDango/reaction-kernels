# Reaction Kernels Lab (Graph Theory)

This repository contains the setup and helper code for the **Reaction Kernels Lab**.
The goal of the project is to represent **chemical reactions as graphs** and apply
**graph kernel methods** for **reaction classification** using machine learning.

This repository is structured for clarity and reproducibility and follows best
practices for Python-based research projects.

---

## Project Structure
reaction-kernels/
├── data/          # datasets (ignored by git, local only)
├── scripts/       # provided helper scripts (e.g. svm_dummy, chem_graph_handling)
├── src/           # own implementations (later work packages)
├── notebooks/     # experiments and analysis
├── .gitignore
└── README.md

---

## Setup (WP0)

### Python Version
- **Python >= 3.11** is required (mandatory for `synkit`)

### Virtual Environment
bash

python3.11 -m venv .venv

source .venv/bin/activate

## Required Libraries
python -m pip install synkit networkx scikit-learn pandas numpy matplotlib

## Dataset Handling
	•	Datasets are stored locally in the data/ folder
	•	Datasets are not tracked by git
	•	Only code and notebooks are uploaded to GitHub

## Definitions (from the project description)
### Graph
A graph ( G = (V, E) ) consists of:
	•	a set of vertices (nodes) ( V )
	•	a set of edges ( E \subseteq V \times V )

In this project, graphs are used to represent molecules and chemical reactions.

### Kernel
A kernel function ( k(x, y) ) computes a similarity measure between two objects.
In this lab, kernels measure similarity between graphs.

### Graph Kernel
A graph kernel is a kernel function defined on graphs.
It allows graphs to be used with standard machine learning algorithms such as
Support Vector Machines (SVMs).

### Weisfeiler–Lehman (WL) Algorithm
The Weisfeiler–Lehman algorithm is an iterative procedure that:
	1.	Assigns initial labels to nodes
	2.	Repeatedly updates each node label based on its neighbors
	3.	Produces a sequence of relabeled graphs ( G_0, G_1, …, G_h )

These relabeled graphs are used to extract increasingly expressive features.

### Weisfeiler–Lehman Graph Kernel
The WL graph kernel computes similarity between two graphs by:
	•	applying a base graph kernel at each WL iteration
	•	summing the similarities over all iterations

Formally:
[
k_{WL}^{(h)}(G, H) = \sum_{i=0}^{h} k(G_i, H_i)
]

### Base Kernels
The project uses different base kernels, including:
	•	Vertex (Subtree) Kernel
Counts matching node labels between two graphs.
	•	Edge Kernel
Counts matching labeled edges (including endpoint labels).
	•	Shortest-Path Kernel
Counts matching shortest paths between pairs of nodes.

### Chemical Reaction Representation
#### Imaginary Transition State (ITS)
	•	Represents a reaction as a single graph
	•	Requires atom-to-atom mapping
	•	Bond changes are encoded directly in edge labels

#### Differential Reaction Fingerprint (DRF)
	•	Does not require atom mapping
	•	Reaction features are computed separately for reactants and products
	•	The final representation is the symmetric difference of these features

### Support Vector Machine (SVM)
An SVM is a supervised learning model that:
	•	uses a kernel function to operate in high-dimensional feature space
	•	is used in this project to classify chemical reactions

## Helper Scripts
	•	chem_graph_handling.py
Utilities for converting reactions and molecules into graph representations.
	•	svm_dummy.py
Minimal example showing how graph kernels can be used with an SVM.

---
## Work Package 1: Data Preparation and Graph Visualization

In Work Package 1, the cleaned reaction dataset was loaded and prepared for further analysis.
The dataset was split into small, class-balanced subsets to enable efficient experimentation
and manual inspection. Each subset contains three reaction classes with twenty reactions per
class, following the project guidelines.

To ensure the correctness of the reaction representations, several visualization utilities
were implemented. Chemical reactions were represented using three complementary graph views:
reactant graphs (educts), product graphs, and the Imaginary Transition State (ITS) graph.
Reactant and product graphs show the molecular structures before and after the reaction,
respectively, while the ITS graph combines both sides and explicitly encodes bond changes
occurring during the reaction.

These visualizations were used as manual sanity checks to verify that reactions were correctly
parsed and converted into graph representations. By inspecting selected examples from each
subset, it was confirmed that atom labels, bond types, and reaction changes were represented
consistently. This step ensures that the subsequent feature extraction and kernel-based
methods operate on meaningful and correct graph structures.

---
## Work Package 2: Feature Extraction with WL and DRF

In Work Package 2, reaction graphs were transformed into structured feature representations using the Weisfeiler–Lehman (WL) framework. Vertex, edge, and shortest-path features were extracted across multiple WL iterations to capture increasingly rich structural context.

Two complementary reaction representations were implemented. For ITS–WL, features were computed directly on the Imaginary Transition State graph and aggregated over all WL iterations. For DRF–WL, features were computed separately for reactant and product graphs and combined using a symmetric difference, ensuring that only reaction-specific changes were retained.

All feature labels were hashed to enable efficient comparison, and the union over all WL iterations formed the final reaction fingerprint. The resulting feature representations were pre-computed and stored for efficient kernel-based classification in subsequent work packages.

---
## Lessons Learned

### Feature Design and Representation
The performance of graph-based kernels strongly depends on the chosen feature representation. Using only atom types as node labels proved insufficient, as many reactions differ primarily in hydrogen count, charge, or aromaticity rather than in atom identity alone. Extending node labels with chemically meaningful attributes such as hydrogen count, charge, and aromaticity was therefore essential to capture relevant reaction changes and to avoid empty DRF feature representations.

Reaction-based representations such as DRF are inherently sparse, as they deliberately discard static molecular structure and retain only features corresponding to bond changes. Consequently, many reaction pairs exhibit zero similarity, which is an expected and desirable property for reaction-centered kernel methods.

### Kernel Construction and Modularity
The Weisfeiler–Lehman procedure substantially increases the expressiveness of graph features by incorporating local neighborhood information across iterations. While higher WL depths improve discriminative power, they also reduce feature overlap between reactions, making the choice of WL iteration depth a critical design parameter.

Representing features as multisets rather than simple sets allows the kernel to account for feature multiplicities. In this project, a multiset intersection kernel was used to preserve the frequency of recurring structural patterns, which is particularly important for reactions involving repeated bond changes.

Separating data loading, feature extraction, kernel computation, and classification into independent modules significantly simplified debugging and experimentation. This modular design enabled systematic comparison of different kernel variants and representations.

### Further Observations
Additional insights gained during extended experimentation and classification experiments will be documented in this section as part of future work.
---

## Notes
This repository covers WP0 (setup and familiarization).
---
