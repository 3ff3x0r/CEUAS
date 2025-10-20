# Statistical Analysis of Perceptions of Brazilian IACUC Members

**Version:** 2.3
**Last Updated:** 2025-10-21

## 1. Project Overview

This repository contains the complete analytical codebase for a post-doctoral research project investigating the perceptions of members of Brazilian Institutional Animal Care and Use Committees (CEUAs - Comissões de Ética no Uso de Animais).

Grounded in the context of Brazilian Law No. 11.794/2008, which mandates the ethical review of all animal use and the inclusion of representatives from animal protection societies (SPAs), this study aims to understand the extent to which CEUAs in Brazil effectively fulfil their ethical mandate to safeguard animal interests in research and education.

The analysis is based on a structured, nationwide web-based survey disseminated by CONCEA to all registered CEUAs in 2020–2021. From this, 369 valid responses were analyzed to provide a national overview and to test recurrent observations reported by committee members. The findings reveal a system structurally constrained by committee composition and institutional dynamics, highlighting a potential gap between the law's intent and its practical implementation. This codebase represents the complete statistical workup of that data.

### Research Team & Affiliations

* Karynn Capilé¹
* Isadora de Castro Travnik¹
* **Erickson Leon Kovalski² (Statistical Analyst)**
* Mônica Ferreira Corrêa³
* Carla Forte Maiolino Molento¹

---

¹ Animal Welfare Laboratory (LABEA), Federal University of Paraná, Curitiba, Paraná – Brazil  
² Data Science, Machine Learning, and Optimization Research Group (CiDAMO), Federal University of Paraná, Curitiba, Paraná – Brazil  
³ Science, Technology and Society Study Group, State University of Rio de Janeiro, Rio de Janeiro – Brazil

## 2. Methodology & Analytical Framework

The project is executed within a Jupyter Notebook environment, leveraging Python and a suite of statistical and visualization libraries. Our methodology is built on a foundation of transparency, reproducibility, and a rigorous commitment to selecting the appropriate statistical tool for the nature of the data.

### 2.1. Data Infrastructure

To ensure data integrity and facilitate complex, reproducible queries, the raw survey data was cleaned and migrated from its original spreadsheet format into a structured **SQLite database** (`ceua_analysis_v3.db`). This relational database, which separates respondent data from survey answers and links codified qualitative responses to lookup tables, serves as the single source of truth for all analyses.

### 2.2. The Three-Phase Analytical Workflow

Our research progressed through three distinct, complementary phases:

**Phase 1: Foundational Univariate Analysis**
A systematic, question-by-question descriptive analysis of each variable to establish a baseline understanding of the sample. This involved a strict protocol of data cleaning, translation (from Portuguese to English), statistical summary, and visualization using a standardized plot format.

**Phase 2: Bivariate Hypothesis Testing**
This phase moved from description to inference, testing specific relationships between pairs of variables. The choice of methodology was strictly governed by the statistical nature of the data, as codified in our `analysis_procedure.md` document:

* **Categorical vs. Categorical:** Chi-Squared Test with Grouped/Stacked Bar Charts.
* **Ordinal vs. Ordinal:** Spearman's Rank Correlation with a hybrid visualization approach.
* **Categorical vs. Ordinal:** Kruskal-Wallis / Mann-Whitney U Test with Box Plots.

**Phase 3: Qualitative Coding and Data Integrity**
A significant portion of our work involved the rigorous processing of unstructured text data.

* **Qualitative Coding:** We established and iteratively refined a formal process for classifying open-ended responses (e.g., Q25, Q45, Q47), involving an inter-rater reliability workflow to produce a robust, coded dataset.
* **Data Integrity Audits:** We created specific diagnostic scripts to perform logical consistency checks on the data (e.g., comparing Q15 vs. Q23), ensuring the validity of variables before their use in hypothesis testing.

## 3. Repository Structure
```
├── Ceuas_stat.ipynb
├── Ceuas_stat.html
├── ceua_analysis_v3.db
├── form1.pdf
├── Codificacoes
├── detailed_similarity_report.html
├── push_to_git.sh
├── similar_rows_report.csv
└── README.md
```
## 4. How to Reproduce the Analysis

The analysis is designed to be fully reproducible.

### 4.1. Prerequisites

* A Python 3.10+ environment (Anaconda is recommended).
* The required Python libraries.

### 4.2. Running the Analysis

1. Clone this repository to your local machine.
2. Ensure all files are in the same directory.
3. Launch Jupyter Notebook or JupyterLab.
4. Open the `Ceuas_stat.ipynb` file.
5. Execute the cells in order to run the full analysis pipeline, from data loading to statistical testing and visualization.
