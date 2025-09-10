import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="COVID-19 Data Analysis System",
    page_icon="🦠",
    layout="wide"
)

# -------------------------------
# Utilities
# -------------------------------
def _slug(s: str) -> str:
    """Normalize column names: lowercase, strip, replace non-alphanum with underscore."""
    s = s.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    return s

def _safe_to_numeric(s):
    try:
        return pd.to_numeric(s)
    except Exception:
        return pd.to_numeric(pd.Series(s), errors="coerce")

def parse_age(value) -> float:
    """Parse messy age values to numeric years. Handles formats like:
    '35', '35-40', '60s', '1 month', '2 months', '0.5', '70+', '25 years'."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()

    # months to years
    m = re.search(r"(\d+\.?\d*)\s*(month|months|mo)\b", s)
    if m:
        return float(m.group(1)) / 12.0

    # weeks to years
    w = re.search(r"(\d+\.?\d*)\s*(week|weeks|wk|wks)\b", s)
    if w:
        return float(w.group(1)) / 52.0

    # 70+, 80+, etc.
    plus = re.match(r"(\d+)\+", s)
    if plus:
        return float(plus.group(1))

    # 60s, 30s
    decade = re.match(r"(\d+)\s*s\b", s)
    if decade:
        return float(decade.group(1))

    # Ranges like 35-40 or 35–40
    r = re.split(r"[-–—]", s)
    if len(r) == 2 and r[0].strip().replace(".", "", 1).isdigit() and r[1].strip().replace(".", "", 1).isdigit():
        a, b = float(r[0]), float(r[1])
        return (a + b) / 2.0

    # Extract first number
    num = re.findall(r"\d+\.?\d*", s)
    if len(num) > 0:
        return float(num[0])

    return np.nan

def parse_yes_no(value) -> int:
    """Parse death/recovered column values into 0/1."""
    if pd.isna(value):
        return 0
    s = str(value).strip().lower()
    yes_tokens = {"yes", "y", "true", "1", "death", "died", "deceased", "dead", "positive"}
    no_tokens = {"no", "n", "false", "0", "alive", "negative"}
    if s in yes_tokens:
        return 1
    if s in no_tokens:
        return 0
    # Fallback: contains any yes token?
    for t in yes_tokens:
        if t in s:
            return 1
    return 0

def standardize_gender(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    s = str(value).strip().lower()
    if s in {"m", "male"}:
        return "Male"
    if s in {"f", "female"}:
        return "Female"
    return "Unknown"

# -------------------------------
# Core Analyzer
# -------------------------------
class CovidDataAnalyzer:
    def __init__(self):
        self.df_covid_data_raw = None
        self.df_line_list_raw = None

        self.covid_df = None            # cleaned covid_19_data
        self.line_df = None             # cleaned line list
        self.country_latest = None      # country totals at their latest available date
        self.country_cluster_df = None  # clustering result

    def load_data(self, covid_file=None, line_list_file=None):
        """Load CSVs from uploader or local filenames. Returns True/False."""
        try:
            if covid_file is None:
                self.df_covid_data_raw = pd.read_csv("covid_19_data.csv")
            else:
                self.df_covid_data_raw = pd.read_csv(covid_file)
        except Exception as e:
            st.error(f"Failed to load covid_19_data.csv: {e}")
            return False

        try:
            if line_list_file is None:
                # Try with common names
                try:
                    self.df_line_list_raw = pd.read_csv("covid19_line_list_data_modified.csv")
                except Exception:
                    self.df_line_list_raw = pd.read_csv("covid19_line_list_data_modified")
            else:
                self.df_line_list_raw = pd.read_csv(line_list_file)
        except Exception as e:
            st.error(f"Failed to load line list data: {e}")
            return False

        return True

    def _standardize_covid_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Map many variants to expected names
        col_map = {}
        for c in df.columns:
            s = _slug(c)
            if s in {"sno", "s_no", "s_no_"}:
                col_map[c] = "SNo"
            elif s in {"observationdate", "observation_date", "date"}:
                col_map[c] = "ObservationDate"
            elif s in {"province_state", "provincestate", "state_province", "province", "state"}:
                col_map[c] = "Province/State"
            elif s in {"country_region", "country", "country__region"}:
                col_map[c] = "Country/Region"
            elif s in {"last_update", "lastupdate", "last_updated"}:
                col_map[c] = "Last Update"
            elif s in {"confirmed"}:
                col_map[c] = "Confirmed"
            elif s in {"deaths", "death"}:
                col_map[c] = "Deaths"
            elif s in {"recovered"}:
                col_map[c] = "Recovered"
        return df.rename(columns=col_map)

    def _standardize_line_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {}
        for c in df.columns:
            s = _slug(c)
            if s in {"id"}:
                col_map[c] = "id"
            elif s in {"case_in_country", "case_incountry", "case"}:
                col_map[c] = "case_in_country"
            elif s in {"reporting_date", "reportingdate", "report_date"}:
                col_map[c] = "reporting date"
            elif s in {"summary"}:
                col_map[c] = "summary"
            elif s in {"location"}:
                col_map[c] = "location"
            elif s in {"country"}:
                col_map[c] = "country"
            elif s in {"gender", "sex"}:
                col_map[c] = "gender"
            elif s in {"age"}:
                col_map[c] = "age"
            elif s in {"symptom_onset", "symptomonset"}:
                col_map[c] = "symptom_onset"
            elif s in {"if_onset_approximated", "ifonsetapproximated"}:
                col_map[c] = "If_onset_approximated"
            elif s in {"hosp_visit_date", "hospital_visit_date"}:
                col_map[c] = "hosp_visit_date"
            elif s in {"exposure_start"}:
                col_map[c] = "exposure_start"
            elif s in {"exposure_end"}:
                col_map[c] = "exposure_end"
            elif s in {"visiting_wuhan"}:
                col_map[c] = "visiting Wuhan"
            elif s in {"from_wuhan"}:
                col_map[c] = "from Wuhan"
            elif s in {"death", "died"}:
                col_map[c] = "death"
            elif s in {"recovered", "recovery"}:
                col_map[c] = "recovered"
            elif s in {"symptom", "symptoms"}:
                col_map[c] = "symptom"
        return df.rename(columns=col_map)

    def clean_data(self):
        # Clean covid data
        dfc = self._standardize_covid_columns(self.df_covid_data_raw.copy())

        # Ensure required columns exist
        needed = {"ObservationDate", "Country/Region", "Confirmed", "Deaths", "Recovered"}
        missing = needed - set(dfc.columns)
        if missing:
            raise ValueError(f"COVID data missing columns: {missing}")

        # Parse dates
        dfc["ObservationDate"] = pd.to_datetime(dfc["ObservationDate"], errors="coerce")
        if "Last Update" in dfc.columns:
            dfc["Last Update"] = pd.to_datetime(dfc["Last Update"], errors="coerce")

        # Fill and enforce numeric types
        for c in ["Confirmed", "Deaths", "Recovered"]:
            if c in dfc.columns:
                dfc[c] = _safe_to_numeric(dfc[c]).fillna(0).astype(int)

        if "Province/State" not in dfc.columns:
            dfc["Province/State"] = "Unknown"

        dfc["Province/State"] = dfc["Province/State"].fillna("Unknown")
        dfc["Country/Region"] = dfc["Country/Region"].fillna("Unknown")

        # Keep cleaned
        self.covid_df = dfc

        # Build country latest totals (sum provinces at each country's latest date)
        latest_by_country = (
            self.covid_df.groupby("Country/Region")["ObservationDate"].max().reset_index()
            .rename(columns={"ObservationDate": "LatestDate"})
        )
        merged = self.covid_df.merge(latest_by_country, on="Country/Region", how="left")
        latest_rows = merged[merged["ObservationDate"] == merged["LatestDate"]]
        self.country_latest = (
            latest_rows.groupby("Country/Region", as_index=False)[["Confirmed", "Deaths", "Recovered"]].sum()
        )

        # Clean line list data
        dfl = self._standardize_line_columns(self.df_line_list_raw.copy())

        # Basic fills
        for c in ["gender", "country", "location", "summary", "symptom"]:
            if c in dfl.columns:
                dfl[c] = dfl[c].fillna("Unknown")

        # Parse dates
        for c in ["reporting date", "symptom_onset", "hosp_visit_date", "exposure_start", "exposure_end"]:
            if c in dfl.columns:
                dfl[c] = pd.to_datetime(dfl[c], errors="coerce")

        # Parse age
        if "age" in dfl.columns:
            dfl["age_numeric"] = dfl["age"].apply(parse_age)
        else:
            dfl["age_numeric"] = np.nan

        # Standardize gender
        if "gender" in dfl.columns:
            dfl["gender"] = dfl["gender"].apply(standardize_gender)
        else:
            dfl["gender"] = "Unknown"

        # Create age groups
        bins = [0, 18, 30, 50, 70, 120]
        labels = ["0-17", "18-29", "30-49", "50-69", "70+"]
        dfl["age_group"] = pd.cut(dfl["age_numeric"], bins=bins, labels=labels, include_lowest=True)

        # Parse death and recovered to binary where present
        if "death" in dfl.columns:
            dfl["death_binary"] = dfl["death"].apply(parse_yes_no).astype(int)
        else:
            dfl["death_binary"] = 0
        if "recovered" in dfl.columns:
            dfl["recovered_binary"] = dfl["recovered"].apply(parse_yes_no).astype(int)
        else:
            dfl["recovered_binary"] = 0

        self.line_df = dfl

    def perform_clustering(self, n_clusters=4):
        """KMeans clustering on country-level latest totals with engineered rates."""
        if self.country_latest is None or self.country_latest.empty:
            raise ValueError("Country latest totals not available. Clean data first.")

        df = self.country_latest.copy()
        # Rates (avoid division by zero)
        df["Mortality_Rate"] = np.where(df["Confirmed"] > 0, df["Deaths"] / df["Confirmed"], 0.0)
        df["Recovery_Rate"] = np.where(df["Confirmed"] > 0, df["Recovered"] / df["Confirmed"], 0.0)

        features = ["Confirmed", "Deaths", "Recovered", "Mortality_Rate", "Recovery_Rate"]
        X = df[features].fillna(0.0).astype(float).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        # PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        df["PCA1"] = X_pca[:, 0]
        df["PCA2"] = X_pca[:, 1]

        self.country_cluster_df = df
        return df, kmeans, scaler

    def highest_affected(self):
        """Return highest and second highest affected areas (by Confirmed)."""
        if self.country_latest is None or self.country_latest.empty:
            return None
        ranked = self.country_latest.sort_values("Confirmed", ascending=False).reset_index(drop=True)
        if len(ranked) == 0:
            return None
        res = {"highest": None, "second_highest": None}
        res["highest"] = ranked.iloc[0].to_dict()
        if len(ranked) > 1:
            res["second_highest"] = ranked.iloc[1].to_dict()
        return res

    def mortality_vs_recovery_ratio(self):
        """Compute mortality vs recovery ratio using latest global totals."""
        if self.covid_df is None or self.covid_df.empty:
            return None
        ts = (
            self.covid_df.groupby("ObservationDate")[["Confirmed", "Deaths", "Recovered"]]
            .sum()
            .reset_index()
            .sort_values("ObservationDate")
        )
        latest = ts.iloc[-1]
        total_deaths = int(latest["Deaths"])
        total_recovered = int(latest["Recovered"])
        ratio = (total_deaths / total_recovered) if total_recovered > 0 else np.inf
        return {
            "date": latest["ObservationDate"],
            "total_confirmed": int(latest["Confirmed"]),
            "total_deaths": total_deaths,
            "total_recovered": total_recovered,
            "mortality_recovery_ratio": ratio
        }

    def demographics_summary(self):
        """Distribution by gender and age."""
        if self.line_df is None or self.line_df.empty:
            return {}
        res = {}
        if "gender" in self.line_df.columns:
            res["gender_distribution"] = self.line_df["gender"].value_counts(dropna=False).to_dict()
        if "age_numeric" in self.line_df.columns:
            age_stats = self.line_df["age_numeric"].describe()
            res["age_statistics"] = {
                "count": float(age_stats.get("count", 0)),
                "mean": float(age_stats.get("mean", np.nan)),
                "median": float(self.line_df["age_numeric"].median()),
                "std": float(age_stats.get("std", np.nan)),
                "min": float(age_stats.get("min", np.nan)),
                "max": float(age_stats.get("max", np.nan)),
            }
        if "age_group" in self.line_df.columns:
            res["age_group_distribution"] = self.line_df["age_group"].value_counts().to_dict()
        return res

    def mortality_by_age_group(self):
        """Mortality rate among different age groups (from line list)."""
        if self.line_df is None or self.line_df.empty or "age_group" not in self.line_df.columns:
            return pd.DataFrame()
        df = self.line_df.copy()
        # Drop rows with no age group
        df = df.dropna(subset=["age_group"])
        if df.empty:
            return pd.DataFrame()
        grp = df.groupby("age_group").agg(
            deaths=("death_binary", "sum"),
            total=("death_binary", "count")
        ).reset_index()
        grp["mortality_rate"] = np.where(grp["total"] > 0, 100.0 * grp["deaths"] / grp["total"], 0.0)
        # Ensure categorical order
        order = ["0-17", "18-29", "30-49", "50-69", "70+"]
        grp["age_group"] = pd.Categorical(grp["age_group"], categories=order, ordered=True)
        grp = grp.sort_values("age_group")
        return grp

# -------------------------------
# Session State Initialization
# -------------------------------
if "analyzer" not in st.session_state:
    st.session_state.analyzer = CovidDataAnalyzer()
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "data_cleaned" not in st.session_state:
    st.session_state.data_cleaned = False
if "clustering_done" not in st.session_state:
    st.session_state.clustering_done = False
if "n_clusters" not in st.session_state:
    st.session_state.n_clusters = 4

# -------------------------------
# UI: Header and Navigation
# -------------------------------
st.title("🦠 COVID-19 Data Analysis System")
st.caption("Data Mining & Analytics: Cleaning, Clustering, and Insights")
st.markdown("---")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["📊 Data Loading & Cleaning", "🔬 Clustering Analysis", "📈 Query Results", "📉 Visualizations"],
        index=0
    )
    st.markdown("---")
    st.markdown("How to Use:")
    st.markdown("1) Upload both CSVs")
    st.markdown("2) Load and Clean Data")
    st.markdown("3) Run Clustering")
    st.markdown("4) Explore Queries and Visuals")

# -------------------------------
# Page 1: Data Loading & Cleaning
# -------------------------------
if page == "📊 Data Loading & Cleaning":
    st.header("Data Loading & Cleaning")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload CSV Files")
        covid_file = st.file_uploader("Upload covid_19_data.csv", type=["csv"], key="covid_upl")
        line_file = st.file_uploader("Upload covid19_line_list_data_modified.csv", type=["csv"], key="line_upl")

        if st.button("Load Data", type="primary"):
            ok = st.session_state.analyzer.load_data(covid_file=covid_file, line_list_file=line_file)
            st.session_state.data_loaded = ok
            if ok:
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data. Check files or names.")

    with col2:
        if st.session_state.data_loaded:
            st.subheader("Data Info")
            a = st.session_state.analyzer
            st.write(f"COVID data shape: {a.df_covid_data_raw.shape}")
            st.write(f"Line list shape: {a.df_line_list_raw.shape}")
            if st.button("Clean Data"):
                try:
                    a.clean_data()
                    st.session_state.data_cleaned = True
                    st.success("Data cleaned successfully!")
                except Exception as e:
                    st.session_state.data_cleaned = False
                    st.error(f"Cleaning failed: {e}")

    if st.session_state.data_loaded:
        st.subheader("Preview")
        tab1, tab2 = st.tabs(["COVID-19 Data (raw)", "Line List (raw)"])
        with tab1:
            st.dataframe(st.session_state.analyzer.df_covid_data_raw.head(50))
        with tab2:
            st.dataframe(st.session_state.analyzer.df_line_list_raw.head(50))

    if st.session_state.data_cleaned:
        st.subheader("Cleaned Snapshots")
        tab3, tab4 = st.tabs(["COVID-19 (cleaned)", "Line List (cleaned)"])
        with tab3:
            st.dataframe(st.session_state.analyzer.covid_df.head(50))
        with tab4:
            st.dataframe(st.session_state.analyzer.line_df.head(50))

# -------------------------------
# Page 2: Clustering Analysis
# -------------------------------
elif page == "🔬 Clustering Analysis":
    st.header("K-Means Clustering (Country-level)")

    if not st.session_state.data_cleaned:
        st.warning("Please load and clean data first.")
    else:
        st.session_state.n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4, step=1)

        if st.button("Perform Clustering", type="primary"):
            with st.spinner("Clustering..."):
                df, kmeans, scaler = st.session_state.analyzer.perform_clustering(n_clusters=st.session_state.n_clusters)
                st.session_state.country_cluster_df = df
                st.session_state.clustering_done = True
                st.success("Clustering completed!")

        if st.session_state.clustering_done:
            df = st.session_state.country_cluster_df
            st.subheader("Cluster Overview")

            # Metrics per cluster
            cols = st.columns(min(5, st.session_state.n_clusters))
            for i in range(st.session_state.n_clusters):
                cdf = df[df["Cluster"] == i]
                with cols[i % len(cols)]:
                    st.metric(f"Cluster {i}", f"{len(cdf)} countries")
                    st.caption(f"Avg Confirmed: {cdf['Confirmed'].mean():.0f}")
                    st.caption(f"Avg Deaths: {cdf['Deaths'].mean():.0f}")
                    st.caption(f"Avg Recovery Rate: {cdf['Recovery_Rate'].mean():.2f}")

            # PCA scatter
            fig = px.scatter(
                df,
                x="PCA1",
                y="PCA2",
                color=df["Cluster"].astype(str),
                hover_name="Country/Region",
                hover_data={"Confirmed": True, "Deaths": True, "Recovered": True, "PCA1": False, "PCA2": False},
                title="Country Clusters (PCA Projection)",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Clustered Table")
            st.dataframe(df.sort_values(["Cluster", "Confirmed"], ascending=[True, False]).reset_index(drop=True))

# -------------------------------
# Page 3: Query Results
# -------------------------------
elif page == "📈 Query Results":
    st.header("Answers to Queries")

    if not st.session_state.data_cleaned:
        st.warning("Please load and clean data first.")
    else:
        a = st.session_state.analyzer

        # i) Highest affected areas
        st.subheader("i) Highest and Second-Highest Affected Areas (by Confirmed)")
        hi = a.highest_affected()
        if hi is None:
            st.info("No country totals available.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                h = hi["highest"]
                st.info("Highest Affected Area")
                st.write(f"Country: {h['Country/Region']}")
                st.write(f"Confirmed: {int(h['Confirmed']):,}")
                st.write(f"Deaths: {int(h['Deaths']):,}")
                st.write(f"Recovered: {int(h['Recovered']):,}")
            with col2:
                sh = hi.get("second_highest")
                if sh:
                    st.info("Second Highest Affected Area")
                    st.write(f"Country: {sh['Country/Region']}")
                    st.write(f"Confirmed: {int(sh['Confirmed']):,}")
                    st.write(f"Deaths: {int(sh['Deaths']):,}")
                    st.write(f"Recovered: {int(sh['Recovered']):,}")
                else:
                    st.write("Only one country present in data.")

        # ii) Mortality vs Recovery ratio
        st.subheader("ii) Mortality vs Recovery Ratio (latest global totals)")
        mr = a.mortality_vs_recovery_ratio()
        if mr is None:
            st.info("Not enough data to compute global totals.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Date", mr["date"].date().isoformat() if isinstance(mr["date"], pd.Timestamp) else str(mr["date"]))
            with c2:
                st.metric("Total Confirmed", f"{mr['total_confirmed']:,}")
            with c3:
                st.metric("Total Deaths", f"{mr['total_deaths']:,}")
            with c4:
                st.metric("Total Recovered", f"{mr['total_recovered']:,}")

            ratio_text = f"{mr['mortality_recovery_ratio']:.4f}" if np.isfinite(mr["mortality_recovery_ratio"]) else "∞"
            st.metric("Mortality/Recovery Ratio", ratio_text)

        # iii) Demographic tendency
        st.subheader("iii) Demographic Tendency (Age, Gender)")
        demo = a.demographics_summary()
        if not demo:
            st.info("Line list data not available.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if "gender_distribution" in demo:
                    st.write("Gender Distribution")
                    gdf = pd.DataFrame(list(demo["gender_distribution"].items()), columns=["Gender", "Count"])
                    fig = px.pie(gdf, values="Count", names="Gender", title="Gender Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if "age_statistics" in demo and demo["age_statistics"]["count"] > 0:
                    st.write("Age Statistics")
                    age_stats = demo["age_statistics"]
                    st.write(f"Mean: {age_stats['mean']:.1f}")
                    st.write(f"Median: {age_stats['median']:.1f}")
                    st.write(f"Std: {age_stats['std']:.1f}")
                    st.write(f"Range: {age_stats['min']:.1f} - {age_stats['max']:.1f}")

            if "age_group_distribution" in demo and len(demo["age_group_distribution"]) > 0:
                st.write("Cases by Age Group")
                agdf = pd.DataFrame(list(demo["age_group_distribution"].items()), columns=["Age Group", "Count"])
                # Keep consistent order
                order = ["0-17", "18-29", "30-49", "50-69", "70+"]
                agdf["Age Group"] = pd.Categorical(agdf["Age Group"], categories=order, ordered=True)
                agdf = agdf.sort_values("Age Group")
                fig = px.bar(agdf, x="Age Group", y="Count", title="Age Group Distribution", text="Count")
                st.plotly_chart(fig, use_container_width=True)

        # iv) Mortality rate among different age groups
        st.subheader("iv) Mortality Rate by Age Group")
        m_age = a.mortality_by_age_group()
        if m_age is None or m_age.empty:
            st.info("Mortality-by-age data not available. Ensure 'death' and 'age' fields exist and are parsed.")
        else:
            fig = px.bar(
                m_age,
                x="age_group",
                y="mortality_rate",
                title="Mortality Rate by Age Group",
                labels={"mortality_rate": "Mortality Rate (%)", "age_group": "Age Group"},
                text=m_age["mortality_rate"].round(2)
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            st.write("Detailed Mortality Statistics")
            show_df = m_age.copy()
            show_df["mortality_rate"] = show_df["mortality_rate"].round(2)
            st.dataframe(show_df.reset_index(drop=True))

# -------------------------------
# Page 4: Visualizations
# -------------------------------
elif page == "📉 Visualizations":
    st.header("Data Visualizations")

    if not st.session_state.data_cleaned:
        st.warning("Please load and clean data first.")
    else:
        a = st.session_state.analyzer

        # Top affected countries
        st.subheader("Top N Most Affected Countries (Latest Available)")
        top_n = st.slider("Select N", min_value=5, max_value=25, value=10, step=1)
        cl = a.country_latest.copy().sort_values("Confirmed", ascending=False).head(top_n)
        fig = px.bar(
            cl,
            x="Country/Region",
            y=["Confirmed", "Deaths", "Recovered"],
            title=f"Top {top_n} Countries by COVID-19 Cases",
            labels={"value": "Number of Cases", "variable": "Metric", "Country/Region": "Country"},
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Time series analysis
        st.subheader("Global Trends Over Time")
        ts = (
            a.covid_df.groupby("ObservationDate")[["Confirmed", "Deaths", "Recovered"]]
            .sum()
            .reset_index()
            .sort_values("ObservationDate")
        )
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Confirmed"], mode="lines", name="Confirmed", line=dict(color="blue")))
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Deaths"], mode="lines", name="Deaths", line=dict(color="red")))
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Recovered"], mode="lines", name="Recovered", line=dict(color="green")))
        fig_ts.update_layout(
            title="Global COVID-19 Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Cases",
            hovermode="x unified"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Geographic distribution (requires clustering or at least country_latest)
        st.subheader("Geographic Distribution")
        fig_map = px.choropleth(
            a.country_latest,
            locations="Country/Region",
            locationmode="country names",
            color="Confirmed",
            hover_data=["Deaths", "Recovered"],
            color_continuous_scale="Reds",
            title="COVID-19 Cases by Country (Latest)"
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Clustering map if available
        if st.session_state.get("clustering_done", False) and "country_cluster_df" in st.session_state:
            st.subheader("Clusters by Country")
            dfc = st.session_state.country_cluster_df.copy()
            fig_map2 = px.choropleth(
                dfc,
                locations="Country/Region",
                locationmode="country names",
                color=dfc["Cluster"].astype(str),
                hover_data=["Confirmed", "Deaths", "Recovered", "Cluster"],
                title="Cluster Assignment by Country"
            )
            st.plotly_chart(fig_map2, use_container_width=True)

            st.subheader("Cluster Characteristics")
            cluster_summary = (
                dfc.groupby("Cluster")
                .agg(
                    countries=("Country/Region", "count"),
                    confirmed_mean=("Confirmed", "mean"),
                    deaths_mean=("Deaths", "mean"),
                    recovered_mean=("Recovered", "mean"),
                    mortality_rate_mean=("Mortality_Rate", "mean"),
                    recovery_rate_mean=("Recovery_Rate", "mean"),
                )
                .round(2)
                .reset_index()
            )
            st.dataframe(cluster_summary)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit • K-Means Clustering • Plotly Visualizations")
