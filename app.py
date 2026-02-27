"""
Stack Overflow 2018 Developer Survey — Salary Prediction & Skill Clustering
Streamlit Dashboard + Prediction Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import warnings
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Developer Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# DATA LOADING & PREPROCESSING (cached)
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_and_preprocess():
    """Load CSV, clean, engineer features — mirrors the notebook exactly."""
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    public_csv = os.path.join(DATA_DIR, "survey_results_public.csv")

    # If CSVs are in root (legacy), also check there
    if not os.path.exists(public_csv):
        alt = os.path.join(os.path.dirname(__file__), "survey_results_public.csv")
        if os.path.exists(alt):
            public_csv = alt
        else:
            # Try downloading via kagglehub
            try:
                import kagglehub
                os.makedirs(DATA_DIR, exist_ok=True)
                path = kagglehub.dataset_download(
                    "stackoverflow/stack-overflow-2018-developer-survey"
                )
                for f in os.listdir(path):
                    if f.endswith(".csv"):
                        shutil.copy2(os.path.join(path, f), DATA_DIR)
                public_csv = os.path.join(DATA_DIR, "survey_results_public.csv")
            except Exception as e:
                st.error(f"Could not find or download dataset: {e}")
                st.stop()

    df = pd.read_csv(public_csv)

    # --- Select columns ---
    cols = [
        "Country", "Employment", "FormalEducation", "UndergradMajor",
        "CompanySize", "DevType", "YearsCoding", "YearsCodingProf",
        "JobSatisfaction", "CareerSatisfaction", "ConvertedSalary",
        "LanguageWorkedWith", "FrameworkWorkedWith", "DatabaseWorkedWith",
        "PlatformWorkedWith", "Gender", "Age",
    ]
    df_clean = df[cols].copy()

    # Drop missing / zero salary
    df_clean = df_clean.dropna(subset=["ConvertedSalary"])
    df_clean = df_clean[df_clean["ConvertedSalary"] > 0]

    # Remove outliers (1st–99th percentile)
    q1 = df_clean["ConvertedSalary"].quantile(0.01)
    q99 = df_clean["ConvertedSalary"].quantile(0.99)
    df_clean = df_clean[
        (df_clean["ConvertedSalary"] >= q1) & (df_clean["ConvertedSalary"] <= q99)
    ]

    # Fill categorical NaN → 'Unknown'
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = df_clean[col].fillna("Unknown")

    # ---- Feature engineering ----
    years_map = {
        "0-2 years": 1, "3-5 years": 4, "6-8 years": 7,
        "9-11 years": 10, "12-14 years": 13, "15-17 years": 16,
        "18-20 years": 19, "21-23 years": 22, "24-26 years": 25,
        "27-29 years": 28, "30 or more years": 32, "Unknown": 0,
    }
    df_clean["YearsCoding_Num"] = df_clean["YearsCoding"].map(years_map).fillna(0)
    df_clean["YearsCodingProf_Num"] = df_clean["YearsCodingProf"].map(years_map).fillna(0)

    satisfaction_map = {
        "Extremely dissatisfied": 1, "Moderately dissatisfied": 2,
        "Slightly dissatisfied": 3, "Neither satisfied nor dissatisfied": 4,
        "Slightly satisfied": 5, "Moderately satisfied": 6,
        "Extremely satisfied": 7, "Unknown": 4,
    }
    df_clean["JobSatisfaction_Num"] = df_clean["JobSatisfaction"].map(satisfaction_map).fillna(4)
    df_clean["CareerSatisfaction_Num"] = df_clean["CareerSatisfaction"].map(satisfaction_map).fillna(4)

    for col_name, src in [
        ("Num_Languages", "LanguageWorkedWith"),
        ("Num_Frameworks", "FrameworkWorkedWith"),
        ("Num_Databases", "DatabaseWorkedWith"),
        ("Num_Platforms", "PlatformWorkedWith"),
    ]:
        df_clean[col_name] = df_clean[src].apply(
            lambda x: len(str(x).split(";")) if x != "Unknown" else 0
        )

    df_clean["Total_Skills"] = (
        df_clean["Num_Languages"]
        + df_clean["Num_Frameworks"]
        + df_clean["Num_Databases"]
        + df_clean["Num_Platforms"]
    )

    edu_order = {
        "I never completed any formal education": 0,
        "Primary/elementary school": 1,
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 2,
        "Some college/university study without earning a degree": 3,
        "Associate degree": 4,
        "Bachelor's degree (BA, BS, B.Eng., etc.)": 5,
        "Master's degree (MA, MS, M.Eng., MBA, etc.)": 6,
        "Professional degree (JD, MD, etc.)": 7,
        "Other doctoral degree (Ph.D, Ed.D., etc.)": 8,
        "Unknown": 3,
    }
    df_clean["Education_Level"] = df_clean["FormalEducation"].map(edu_order).fillna(3)

    return df_clean


@st.cache_resource(show_spinner="Training models…")
def train_models(_df):
    """Train Linear Regression & K-Means (mirrors notebook). Returns models + metrics."""
    df = _df.copy()

    # --- Linear Regression ---
    feature_cols = [
        "YearsCoding_Num", "YearsCodingProf_Num", "Education_Level",
        "Num_Languages", "Num_Frameworks", "Num_Databases",
        "Num_Platforms", "Total_Skills", "JobSatisfaction_Num",
        "CareerSatisfaction_Num",
    ]
    target = "ConvertedSalary"
    df_reg = df[feature_cols + [target]].dropna()

    X = df_reg[feature_cols]
    y = df_reg[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred_test = lr_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, lr_model.predict(X_train))

    # --- K-Means ---
    cluster_features = [
        "YearsCoding_Num", "YearsCodingProf_Num", "Education_Level",
        "Num_Languages", "Num_Frameworks", "Num_Databases",
        "Num_Platforms", "Total_Skills", "ConvertedSalary",
    ]
    df_cluster = df[cluster_features].dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
    df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, kmeans.labels_, sample_size=5000, random_state=42)

    cluster_summary = (
        df_cluster.groupby("Cluster")
        .agg({
            "ConvertedSalary": ["mean", "median", "count"],
            "YearsCodingProf_Num": "mean",
            "Total_Skills": "mean",
            "Education_Level": "mean",
        })
        .round(1)
    )
    cluster_summary.columns = [
        "Avg Salary", "Median Salary", "Count",
        "Avg YearsProf", "Avg TotalSkills", "Avg Education",
    ]

    return {
        "lr_model": lr_model,
        "feature_cols": feature_cols,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "kmeans": kmeans,
        "scaler": scaler,
        "df_cluster": df_cluster,
        "cluster_summary": cluster_summary,
        "sil_score": sil_score,
        "X_scaled": X_scaled,
    }


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
df_clean = load_and_preprocess()
models = train_models(df_clean)

# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📈 EDA Dashboard",
        "🔮 Salary Predictor",
        "👥 Cluster Explorer",
        "📋 Model Performance",
    ],
)

# ══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("💰 Developer Salary Prediction & Skill Clustering")
    st.markdown(
        """
        **Business Problem:** Examine salary trends and skill demand in the global software
        market using regression-based salary prediction and K-Means clustering.

        **Dataset:** Stack Overflow 2018 Developer Survey — 98,855 respondents across 100+ countries.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Developers", f"{len(df_clean):,}")
    col2.metric("Median Salary", f"${df_clean['ConvertedSalary'].median():,.0f}")
    col3.metric("R² Score (Test)", f"{models['test_r2']:.4f}")
    col4.metric("Silhouette Score", f"{models['sil_score']:.4f}")

    st.markdown("---")
    st.subheader("Economic Concepts Applied")
    concepts = {
        "Demand-Supply Dynamics": "Scarce skills (Go, Scala) → salary premium; commodity skills (HTML) → wage pressure",
        "Human Capital Theory": "Experience & education = investments yielding salary returns",
        "Labor Market Segmentation": "K-Means reveals junior / mid / senior / expert tiers",
        "Revenue Optimization": "Cluster-based salary benchmarks optimize eng budgets",
        "Pricing Strategy": "Specialization creates pricing power for developers",
        "Risk Analysis": "Over-reliance on common skills → wage stagnation risk",
    }
    for concept, desc in concepts.items():
        st.markdown(f"- **{concept}:** {desc}")


# ══════════════════════════════════════════════
# PAGE 2 — EDA DASHBOARD
# ══════════════════════════════════════════════
elif page == "📈 EDA Dashboard":
    st.title("📈 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Salary Distribution",
        "Top Countries",
        "Programming Languages",
        "Education vs Salary",
        "Correlation Heatmap",
    ])

    # --- Tab 1: Salary Distribution ---
    with tab1:
        st.subheader("Salary Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(df_clean["ConvertedSalary"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0].axvline(df_clean["ConvertedSalary"].median(), color="red", linestyle="--",
                        label=f"Median: ${df_clean['ConvertedSalary'].median():,.0f}")
        axes[0].set_title("Annual Salary Distribution (USD)")
        axes[0].set_xlabel("Salary (USD)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[1].boxplot(df_clean["ConvertedSalary"], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="lightblue"))
        axes[1].set_title("Salary Box Plot")
        axes[1].set_ylabel("Salary (USD)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Salary", f"${df_clean['ConvertedSalary'].mean():,.0f}")
        col2.metric("Median Salary", f"${df_clean['ConvertedSalary'].median():,.0f}")
        col3.metric("Std Dev", f"${df_clean['ConvertedSalary'].std():,.0f}")

    # --- Tab 2: Top Countries ---
    with tab2:
        st.subheader("Top 15 Countries by Median Developer Salary")
        n_countries = st.slider("Number of countries", 5, 25, 15)
        country_salary = (
            df_clean.groupby("Country")["ConvertedSalary"]
            .agg(["median", "count"])
            .rename(columns={"median": "Median_Salary", "count": "Respondents"})
            .query("Respondents >= 50")
            .sort_values("Median_Salary", ascending=False)
            .head(n_countries)
        )
        fig, ax = plt.subplots(figsize=(12, max(5, n_countries * 0.45)))
        ax.barh(country_salary.index[::-1], country_salary["Median_Salary"][::-1],
                color=plt.cm.viridis(np.linspace(0.3, 0.9, n_countries)), edgecolor="black")
        ax.set_xlabel("Median Annual Salary (USD)")
        ax.set_title(f"Top {n_countries} Countries by Median Salary")
        for i, (val, count) in enumerate(
            zip(country_salary["Median_Salary"][::-1], country_salary["Respondents"][::-1])
        ):
            ax.text(val + 1000, i, f"${val:,.0f} (n={count})", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Tab 3: Languages ---
    with tab3:
        st.subheader("Top Programming Languages by Developer Usage")
        all_languages = df_clean["LanguageWorkedWith"].str.split(";").explode()
        lang_counts = all_languages.value_counts().head(20)
        fig, ax = plt.subplots(figsize=(12, 7))
        lang_counts[::-1].plot(kind="barh", color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, 20)),
                               edgecolor="black", ax=ax)
        ax.set_title("Top 20 Programming Languages")
        ax.set_xlabel("Number of Developers")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Skill premium
        st.subheader("Salary Premium by Language")
        top_langs = lang_counts.head(15).index.tolist()
        premiums = []
        for lang in top_langs:
            knows = df_clean[df_clean["LanguageWorkedWith"].str.contains(lang, na=False)]["ConvertedSalary"].median()
            doesnt = df_clean[~df_clean["LanguageWorkedWith"].str.contains(lang, na=False)]["ConvertedSalary"].median()
            premiums.append({"Language": lang, "Premium": knows - doesnt})
        prem_df = pd.DataFrame(premiums).sort_values("Premium")
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#e74c3c" if p < 0 else "#27ae60" for p in prem_df["Premium"]]
        ax.barh(prem_df["Language"], prem_df["Premium"], color=colors, edgecolor="black")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_title("Salary Premium: Knowing vs Not Knowing a Language")
        ax.set_xlabel("Premium (USD)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Tab 4: Education ---
    with tab4:
        st.subheader("Median Salary by Education Level")
        edu_salary = (
            df_clean[df_clean["FormalEducation"] != "Unknown"]
            .groupby("FormalEducation")["ConvertedSalary"]
            .median()
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        edu_salary.plot(kind="barh", color="teal", edgecolor="black", ax=ax)
        ax.set_xlabel("Median Annual Salary (USD)")
        ax.set_title("Salary by Education Level")
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Tab 5: Correlation ---
    with tab5:
        st.subheader("Correlation Heatmap")
        num_cols = [
            "ConvertedSalary", "YearsCoding_Num", "YearsCodingProf_Num",
            "JobSatisfaction_Num", "CareerSatisfaction_Num", "Education_Level",
            "Num_Languages", "Num_Frameworks", "Num_Databases", "Num_Platforms",
            "Total_Skills",
        ]
        corr = df_clean[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
# PAGE 3 — SALARY PREDICTOR
# ══════════════════════════════════════════════
elif page == "🔮 Salary Predictor":
    st.title("🔮 Predict Developer Salary")
    st.markdown(
        "Adjust the sliders below to input a developer profile, "
        "and the Linear Regression model will predict the annual salary (USD)."
    )

    col1, col2 = st.columns(2)

    with col1:
        years_coding = st.slider("Years of Coding Experience", 0, 35, 5)
        years_prof = st.slider("Years of Professional Experience", 0, 35, 3)
        education = st.selectbox(
            "Education Level",
            options=[
                ("No formal education", 0),
                ("Primary school", 1),
                ("High school", 2),
                ("Some college (no degree)", 3),
                ("Associate degree", 4),
                ("Bachelor's degree", 5),
                ("Master's degree", 6),
                ("Professional degree (JD, MD)", 7),
                ("Doctoral degree (PhD)", 8),
            ],
            index=5,
            format_func=lambda x: x[0],
        )[1]
        num_languages = st.slider("Number of Programming Languages Known", 0, 20, 4)
        num_frameworks = st.slider("Number of Frameworks Known", 0, 15, 2)

    with col2:
        num_databases = st.slider("Number of Databases Known", 0, 10, 2)
        num_platforms = st.slider("Number of Platforms Known", 0, 10, 2)
        total_skills = num_languages + num_frameworks + num_databases + num_platforms
        st.metric("Total Skills (auto-calculated)", total_skills)
        job_satisfaction = st.slider("Job Satisfaction (1-7)", 1, 7, 5)
        career_satisfaction = st.slider("Career Satisfaction (1-7)", 1, 7, 5)

    # Predict
    input_data = np.array([[
        years_coding, years_prof, education,
        num_languages, num_frameworks, num_databases,
        num_platforms, total_skills,
        job_satisfaction, career_satisfaction,
    ]])

    prediction = models["lr_model"].predict(input_data)[0]

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.success(f"### 💰 Predicted Annual Salary: **${max(0, prediction):,.0f}**")

    # Show feature contributions
    st.subheader("Feature Contributions to Prediction")
    coeffs = models["lr_model"].coef_
    feature_names = models["feature_cols"]
    input_vals = input_data[0]
    contributions = coeffs * input_vals

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": input_vals,
        "Coefficient": coeffs,
        "Contribution (USD)": contributions,
    }).sort_values("Contribution (USD)", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in contrib_df["Contribution (USD)"]]
    ax.barh(contrib_df["Feature"], contrib_df["Contribution (USD)"], color=colors, edgecolor="black")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to Predicted Salary (USD)")
    ax.set_title("How Each Feature Affects This Prediction")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    intercept = models["lr_model"].intercept_
    st.caption(
        f"Base salary (intercept): ${intercept:,.0f} + sum of contributions = ${max(0, prediction):,.0f}"
    )


# ══════════════════════════════════════════════
# PAGE 4 — CLUSTER EXPLORER
# ══════════════════════════════════════════════
elif page == "👥 Cluster Explorer":
    st.title("👥 Developer Skill Clusters (K-Means)")
    st.markdown("K-Means (K=4) segments developers into distinct market tiers.")

    # Cluster summary table
    st.subheader("Cluster Profiles")
    summary = models["cluster_summary"].copy()
    summary.index = [f"Cluster {i}" for i in summary.index]
    st.dataframe(summary.style.format({
        "Avg Salary": "${:,.0f}",
        "Median Salary": "${:,.0f}",
        "Count": "{:,}",
        "Avg YearsProf": "{:.1f}",
        "Avg TotalSkills": "{:.1f}",
        "Avg Education": "{:.1f}",
    }), use_container_width=True)

    st.metric("Silhouette Score", f"{models['sil_score']:.4f}")

    # Visualizations
    df_cluster = models["df_cluster"]
    colors_map = {0: "#2ecc71", 1: "#3498db", 2: "#f1c40f", 3: "#e74c3c"}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Salary vs Professional Experience")
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in range(4):
            mask = df_cluster["Cluster"] == c
            ax.scatter(
                df_cluster.loc[mask, "YearsCodingProf_Num"],
                df_cluster.loc[mask, "ConvertedSalary"],
                alpha=0.3, s=10, color=colors_map[c], label=f"Cluster {c}",
            )
        ax.set_xlabel("Years Coding Professionally")
        ax.set_ylabel("Annual Salary (USD)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Salary vs Total Skills")
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in range(4):
            mask = df_cluster["Cluster"] == c
            ax.scatter(
                df_cluster.loc[mask, "Total_Skills"],
                df_cluster.loc[mask, "ConvertedSalary"],
                alpha=0.3, s=10, color=colors_map[c], label=f"Cluster {c}",
            )
        ax.set_xlabel("Total Technical Skills")
        ax.set_ylabel("Annual Salary (USD)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Cluster size
    st.subheader("Cluster Size Distribution")
    cluster_counts = df_cluster["Cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        [f"Cluster {i}" for i in cluster_counts.index],
        cluster_counts.values,
        color=[colors_map[i] for i in cluster_counts.index],
        edgecolor="black",
    )
    for bar, v in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 50, f"{v:,}", ha="center", fontweight="bold")
    ax.set_ylabel("Number of Developers")
    ax.set_title("Developers per Cluster")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
elif page == "📋 Model Performance":
    st.title("📋 Model Performance Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Linear Regression")
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | R² (Train) | **{models['train_r2']:.4f}** |
        | R² (Test) | **{models['test_r2']:.4f}** |
        | RMSE (Test) | **${models['test_rmse']:,.0f}** |
        """)

        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(models["y_test"], models["y_pred_test"], alpha=0.15, s=8, color="steelblue")
        max_val = max(models["y_test"].max(), models["y_pred_test"].max())
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")
        ax.set_xlabel("Actual Salary (USD)")
        ax.set_ylabel("Predicted Salary (USD)")
        ax.set_title("Actual vs Predicted Salary")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("K-Means Clustering")
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | Clusters (K) | **4** |
        | Silhouette Score | **{models['sil_score']:.4f}** |
        """)

        # Residual distribution
        residuals = models["y_test"] - models["y_pred_test"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(residuals, bins=50, color="coral", edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Residual (Actual − Predicted) USD")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Feature importance
    st.subheader("Feature Importance (Regression Coefficients)")
    coef_df = pd.DataFrame({
        "Feature": models["feature_cols"],
        "Coefficient (USD per unit)": models["lr_model"].coef_,
    }).sort_values("Coefficient (USD per unit)", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in coef_df["Coefficient (USD per unit)"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient (USD per unit)"], color=colors, edgecolor="black")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (USD impact per unit increase)")
    ax.set_title("How Each Feature Affects Salary")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(f"Intercept (base salary): ${models['lr_model'].intercept_:,.0f}")

    st.markdown("---")
    st.subheader("Key Business Takeaways")
    st.markdown("""
    1. **Demand-Supply:** Specialized roles (DevOps, ML Engineers) command salary premiums due to supply shortages.
    2. **Human Capital Theory:** Professional experience is the strongest predictor — each year adds significant earning potential.
    3. **Pricing Strategy:** Cluster profiles enable competitive salary bands for different developer tiers.
    4. **Risk Analysis:** Over-reliance on common skills (JS, HTML) carries wage stagnation risk.
    5. **Revenue Optimization:** Hire from cost-efficient clusters and upskill strategically.
    """)
