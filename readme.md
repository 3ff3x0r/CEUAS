# Perceptions of IACUC Members in Brazil: A Statistical Analysis

## 1\. Project Overview

This repository contains the statistical analysis for a post-doctoral research project conducted at the Universidade Federal do Paraná (UFPR). The project aims to analyze the perceptions of members of Brazilian Institutional Animal Care and Use Committees (CEUAs - Comissões de Ética no Uso de Animais) regarding animal ethics, committee functions, and the welfare of animals used in scientific research.

The analysis is based on an anonymized survey distributed to CEUA members across Brazil. This study seeks to uncover correlations and significant differences in opinion based on demographic, professional, and ethical standpoints.

**Statistical Analyst:** Leon

## 2\. Research Questions

This analysis seeks to answer a series of specific hypotheses concerning the relationships between the members' backgrounds, beliefs, and their roles within the CEUA. Key questions include:

  - Is there an association between the professional **use of animals** and the opinion that animal experimentation is a **"necessary evil"**?
  - Do **vegans/vegetarians** perceive a different level of **animal suffering** in experiments compared to non-vegetarians?
  - Is there a correlation between a member's **formal ethics training** and their self-assessed **aptitude for ethical evaluations**?
  - Is there a relationship between a member's **role on the committee** and whether they feel their **opinions are respected**?
  - Does the **number of rejected proposals** correlate with the **member's tenure** on the committee?
  - How do institutional factors, such as the **nature of the institution** (public vs. private) and the **presence of an SPA representative**, relate to the number of licenses refused?

## 3\. Methodology

The analysis is conducted in a Jupyter Notebook (`Ceuas_stat.ipynb`) using Python. The methodology is designed to be transparent, reproducible, and statistically rigorous.

### 3.1. Data Source

The primary data is a single, anonymized survey dataset collected from CEUA members. The raw data was cleaned, processed, and imported into a structured **SQLite database** (`ceua_analysis_v3.db`) for efficient and reliable querying. The database includes the main survey answers, respondent information, and lookup tables for coded categorical variables.

### 3.2. Statistical Tests

The choice of statistical test is determined by the nature of the variables being compared for each hypothesis. The primary tests used are:

  - **Chi-Square (χ²) Test of Independence:** Used to determine if there is a significant association between two categorical variables (e.g., "User vs. Non-user" and "Favorable vs. Critical Justification").
  - **Mann-Whitney U Test:** A non-parametric test used to compare the distributions of an ordinal variable (e.g., a Likert scale score) between two independent groups (e.g., "Vegans vs. Non-vegans").

All tests are conducted with a significance level (α) of 0.05.

### 3.3. Tools & Libraries

  - **Language:** Python 3.10
  - **Environment:** Jupyter Notebook
  - **Core Libraries:**
      - `pandas`: For data manipulation and structuring.
      - `sqlite3`: For querying the project database.
      - `scipy.stats`: For performing the statistical tests (chi2\_contingency, mannwhitneyu).
      - `matplotlib` & `seaborn`: For data visualization.

## 4\. Repository Structure

```
.
├── data/
│   └── ceua_analysis_v3.db       # The project's SQLite database
├── notebooks/
│   └── Ceuas_stat.ipynb          # Jupyter Notebook with the full analysis
├── reports/
│   └── Ceuas_stat.html           # Exported HTML report (code hidden)
└── README.md                     # This file
```

## 5\. How to Reproduce the Analysis

To ensure scientific validity, the analysis is fully reproducible.

### 5.1. Prerequisites

  - A Python environment (Anaconda is recommended).
  - The required Python libraries installed. You can install them via pip:
    ```bash
    pip install pandas scipy matplotlib seaborn jupyter
    ```

### 5.2. Running the Analysis

1.  Clone this repository to your local machine.
2.  Navigate to the `/notebooks` directory.
3.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter-notebook
    ```
4.  Open the `Ceuas_stat.ipynb` file.
5.  Execute the cells in order to run the full analysis pipeline, from data loading to statistical testing and visualization.

## 6\. License

This research project is licensed under the MIT License. See the `LICENSE` file for details.
