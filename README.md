# 🍷 Supervised Learning: White Wine Quality Prediction

**Contributors:** Khaled, Rudy, Jules

---

## 1. Business Use Case (BUC)
**Context:**
A mid-sized winery produces thousands of liters of white wine daily. Currently, quality assessment relies on human sommeliers at the end of the production line. This process is:
* **Subjective:** Grades vary between tasters.
* **Expensive:** High labor costs for experts.
* **Reactive:** Poor quality is detected only *after* the wine is finished.

**Stakeholders:** Production Managers, Quality Control Lab.

**Goal:**
Automate the prediction of the sensory quality score (0-10) using objective physicochemical lab data available during production.

**Business Value:**
By deploying this model, the winery can:
1.  **Standardize Quality:** Remove human bias from grading.
2.  **Reduce Costs:** Minimize reliance on manual tasting.
3.  **Optimize Production:** Detect low-quality batches early (based on density/sugar levels) and intervene before bottling.

---

## 2. Dataset
We used the **Wine Quality (White)** dataset from the UCI Machine Learning Repository.

* **Source:** `winequality-white.csv` (Included in this repository).
* **Description:** Real-world data containing 4,898 samples of "Vinho Verde" wine variants.
* **Features:** 11 physicochemical properties (e.g., Alcohol, Density, Volatile Acidity).
* **Target:** `quality` (Ordinal score between 3 and 9).

---

## 3. Methodology & Experiment Tracking
We followed a rigorous supervised learning pipeline, adhering to best practices (Train/Test Split, Cross-Validation, and Baselines).

| Experiment | Model | Preprocessing | MSE (Error) | Findings |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | DummyRegressor (Mean) | None | **0.7557** | Established the performance floor. Any useful model must beat this error. |
| **Exp 1** | Linear Regression (OLS) | None | **0.5690** | **~25% improvement** over baseline. Confirmed linear relationships exist. |
| **Exp 2** | Ridge Regression | StandardScaler + Ridge | **0.5690** | **Best Model.** Selected via GridSearch (Best alpha: 0.01). Chosen over OLS because it handles the multicollinearity between Density and Sugar robustly. |

**Key Insight:**
Our EDA and Model interpretation revealed a "Multicollinearity Trap" between **Density** and **Residual Sugar**. The Ridge model successfully identified that while Sugar positively impacts quality, high Density (often caused by sugar) negatively impacts it.

---

## 4. Project Structure
The repository is organized as follows:

```text
├── README.md              # Project documentation and reproduction guide
├── main.py                # End-to-end training pipeline (Load -> Train -> Evaluate)
├── eda.ipynb              # Exploratory Data Analysis (Visualizations & Insights)
└── winequality-white.csv  # The dataset


5. Reproduction Instructions

Follow these steps to reproduce our results exactly

Step 1: Environment Setup

Ensure you have Python 3.8+ installed. Install the required dependencies:
pip install pandas scikit-learn matplotlib seaborn

Step 2: Run the Training Pipeline

Execute the main.py script. This will:
Load the dataset.
Split the data (80% Train / 20% Test).
Train the optimized Ridge Regression model.
Output the Baseline and Final MSE scores.
python main.py

Step 3: Explore the Data

Open eda.ipynb in Jupyter Notebook or Google Colab to view the detailed charts, including:
Target Distribution (Class Imbalance).
Correlation Heatmap.
Feature Importance Analysis.

