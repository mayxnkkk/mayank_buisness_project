# Data Analytics Project — Stack Overflow 2018 Developer Survey Analysis

## Business Problem Statement

**Examine salary trends and skill demand in the global software market using regression-based salary prediction.**

The global software industry faces critical challenges in compensation benchmarking, talent acquisition, and workforce planning. With rapidly evolving technology stacks, companies struggle to set competitive salaries while developers seek to maximize their earning potential through strategic skill development.

This project uses the Stack Overflow 2018 Developer Survey (98,000+ respondents across 100+ countries) to:
- **Predict developer salaries** based on experience, education, and skill profiles
- **Identify distinct developer segments** through unsupervised clustering
- **Analyze demand-supply dynamics** across programming languages and roles
- **Provide actionable insights** for HR strategy, compensation planning, and career development

---

## Economic Concepts Applied

| Concept | Application in This Project |
|---|---|
| **Demand-Supply Dynamics** | Analyzing how developer supply (number of devs with a skill) vs. market demand affects salary levels. Scarce skills (Go, Scala) command premiums while commodity skills (HTML, CSS) face wage pressure. |
| **Human Capital Theory** | Professional experience and education are treated as investments that yield salary returns. The regression model quantifies the ROI per year of experience. |
| **Labor Market Segmentation** | K-Means clustering reveals distinct segments (junior, mid, senior, expert) — each with different compensation norms, similar to segmented labor markets. |
| **Revenue Optimization** | Companies can use cluster-based salary benchmarks to optimize engineering budgets — hiring from efficient segments and upskilling strategically. |
| **Pricing Strategy** | Salary = "price of labor". The analysis reveals how specialization creates pricing power for developers, while generalist oversupply pushes wages toward equilibrium. |
| **Risk Analysis** | Over-reliance on common skills carries wage stagnation risk. Diversifying into niche/emerging technologies serves as a hedge against market saturation. |
| **Marginal Productivity** | Regression coefficients estimate the marginal contribution of each additional skill/year of experience to salary — reflecting marginal productivity of labor. |

---

## AI Techniques Used

### 1. K-Means Clustering (Unsupervised Learning)
- **Purpose:** Segment developers into distinct groups based on skills, experience, and salary
- **Features:** Years of coding, professional experience, education level, number of languages/frameworks/databases/platforms known, total skills, salary
- **Method:** Elbow method + Silhouette score to determine optimal K
- **Outcome:** 4 distinct developer clusters (Entry-Level, Mid-Level, Senior, Expert)

### 2. Linear Regression (Supervised Learning)
- **Purpose:** Predict annual developer salary
- **Features:** Years coding, professional experience, education level, number of languages, frameworks, databases, platforms, total skills, job satisfaction, career satisfaction
- **Target:** Converted Annual Salary (USD)
- **Split:** 80% train / 20% test
- **Metrics:** R² Score, RMSE

### 3. Data Cleaning & Preprocessing

#### How We Handle Missing / Null Values

| Data Issue | Strategy | Rationale |
|---|---|---|
| **Missing salary (`ConvertedSalary`)** | **Delete** rows with null/zero salary | Salary is our prediction target. Imputing it (with mean or median) would fabricate the very value we're trying to predict, introducing circular bias into both the regression model and cluster analysis. |
| **Salary outliers** | **Delete** rows outside 1st–99th percentile | Extreme values (e.g., $1/year or $10M/year) are likely data-entry errors. They skew the mean, distort regression coefficients, and pull K-Means centroids away from meaningful positions. |
| **Missing categorical features** (Country, DevType, Gender, etc.) | **Fill with `'Unknown'`** (a new category) | Deleting every row with any missing categorical value would slash the dataset dramatically (many columns have 10–30% missing). `'Unknown'` preserves the row's other valid data. For multi-value fields like `LanguageWorkedWith`, `'Unknown'` maps to 0 skills — a neutral and interpretable value. |
| **Missing ordinal features** (YearsCoding, Satisfaction, Education) | **Fill with neutral midpoint** (0 for years, 4 for satisfaction, 3 for education) | These are mapped to manual numeric scales (not continuous). A neutral midpoint avoids pulling averages in either direction. This is more semantically meaningful than a blind statistical median for ordinal data. |

> **Why not median imputation?** For the target variable, deletion is the only honest choice. For features, category-aware filling (`'Unknown'` / neutral midpoint) is more interpretable than a generic statistical median — especially since most features here are categorical or ordinal, not truly continuous.

#### Other Preprocessing Steps
- Feature engineering: ordinal encoding for education, numeric mapping for experience ranges, skill counting from multi-value semicolon-separated fields
- Feature standardization (`StandardScaler`) applied before K-Means clustering to ensure equal feature weighting

### 4. Exploratory Data Analysis (EDA)
- Distribution analysis (histograms, box plots)
- Correlation heatmaps
- Grouped aggregations (salary by country, education, role)
- Multi-value field explosion (languages, frameworks)

---

## Dataset

**Source:** [Stack Overflow 2018 Developer Survey](https://insights.stackoverflow.com/survey/2018)
**Kaggle:** [stackoverflow/stack-overflow-2018-developer-survey](https://www.kaggle.com/datasets/stackoverflow/stack-overflow-2018-developer-survey)

- **File:** `survey_results_public.csv` (98,855 respondents, 129 features)
- **Schema:** `survey_results_schema.csv` (column descriptions)
- **Coverage:** Global, 100+ countries
- **Key Fields Used:** Country, Employment, Education, DevType, YearsCoding, ConvertedSalary, LanguageWorkedWith, FrameworkWorkedWith, DatabaseWorkedWith, PlatformWorkedWith

---

## Screenshots of Outputs & Model Results

### Overview Dashboard
![Overview](screenshots/Overview.png)

### Exploratory Data Analysis (EDA)
![EDA Dashboard](screenshots/EDA.png)

### Salary Predictor (Interactive Prediction Interface)
![Salary Predictor](screenshots/SalaryPredictor.png)

### K-Means Clustering (Developer Segmentation)
![K-Means Clusters](screenshots/KMeans.png)

### Model Performance (Linear Regression Metrics & Residuals)
![Model Performance](screenshots/ModelPerformance.png)

---

## Deployment

**🔗 Live App:** [https://anoop-salary-prediction.streamlit.app](https://anoop-salary-prediction.streamlit.app/)

The project is deployed as a **Streamlit** web application with:
- **Interactive EDA Dashboard** — salary distributions, top countries, language analysis, correlation heatmap
- **Salary Prediction Interface** — input developer profile via sliders → get predicted salary with feature contribution breakdown
- **Cluster Explorer** — visualize K-Means developer segments
- **Model Performance** — R², RMSE, actual vs predicted, residual distribution

### Run Locally

```bash
# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## Expected Outcome

1. **Skill Clusters:** 4 distinct developer segments identified via K-Means — each with unique salary ranges, experience levels, and skill breadths. Useful for HR benchmarking and targeted recruitment.

2. **Salary Prediction Model:** Linear Regression model that predicts annual developer salary based on 10 features. Provides coefficient-based interpretation showing the USD impact of each factor on compensation.

3. **Business Insights:**
   - Specialized roles and niche languages command higher salary premiums
   - Professional experience is the strongest salary predictor
   - The labor market shows clear segmentation that companies can leverage for strategic hiring
   - Skill diversification reduces career risk in a rapidly evolving market

---

## How to Run

### Jupyter Notebook
1. Ensure Python 3.x with the following packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `kagglehub`
2. Open `main.ipynb` and run all cells sequentially — the dataset is downloaded automatically from Kaggle

### Streamlit App
```bash
source .venv/bin/activate
streamlit run app.py
```

---

## Project Structure

```
DA/
├── main.ipynb                    # Jupyter notebook (full analysis)
├── app.py                        # Streamlit web app (dashboard + predictor)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Ignores .venv/, data/, __pycache__
├── screenshots/                  # Output screenshots for README
├── data/                         # Dataset CSVs (auto-downloaded from Kaggle)
│   ├── survey_results_public.csv
│   └── survey_results_schema.csv
└── .venv/                        # Virtual environment (not committed)
```
