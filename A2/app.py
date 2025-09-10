import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

# ------------------ Config ------------------
st.set_page_config(page_title="COVID-19 Analysis", page_icon="🦠", layout="wide")
COVID_PATH = "covid_19_data (1).csv"
LINE_PATH = "covid19_line_list_data_modified (1).csv"

# ------------------ Helpers ------------------
def load_data(covid_path=COVID_PATH, line_path=LINE_PATH):
    covid = pd.read_csv(covid_path)
    line = pd.read_csv(line_path)
    return covid, line

def clean_covid(df):
    # Ensure columns exist (per assignment names)
    needed = ["ObservationDate", "Country/Region", "Confirmed", "Deaths", "Recovered"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in covid_19_data.csv: {missing}")

    # Types
    df["ObservationDate"] = pd.to_datetime(df["ObservationDate"], errors="coerce")
    for c in ["Confirmed", "Deaths", "Recovered"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "Province/State" in df.columns:
        df["Province/State"] = df["Province/State"].fillna("Unknown")

    # Latest totals per country (sum provinces on the latest date per country)
    latest_date = df.groupby("Country/Region")["ObservationDate"].max().reset_index()
    latest_date = latest_date.rename(columns={"ObservationDate": "LatestDate"})
    merged = df.merge(latest_date, on="Country/Region", how="left")
    latest_rows = merged[merged["ObservationDate"] == merged["LatestDate"]]

    country_latest = (
        latest_rows.groupby("Country/Region", as_index=False)[["Confirmed", "Deaths", "Recovered"]].sum()
    )
    return df, country_latest

def clean_line_list(df):
    # Basic fields
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("Unknown").astype(str).str.title()
        df.loc[~df["gender"].isin(["Male", "Female"]), "gender"] = "Unknown"
    else:
        df["gender"] = "Unknown"

    # Age numeric: extract first number (simple)
    if "age" in df.columns:
        df["age_numeric"] = (
            df["age"]
            .astype(str)
            .str.extract(r"(\d+\.?\d*)")[0]
            .astype(float)
        )
    else:
        df["age_numeric"] = np.nan

    # Age groups
    bins = [0, 18, 30, 50, 70, 120]
    labels = ["0-17", "18-29", "30-49", "50-69", "70+"]
    df["age_group"] = pd.cut(df["age_numeric"], bins=bins, labels=labels, include_lowest=True)

    # Death flag (very simple parsing)
    if "death" in df.columns:
        s = df["death"].astype(str).str.lower()
        df["death_binary"] = np.where(
            s.isin(["1", "yes", "true", "death", "died", "deceased"]), 1, 0
        )
    else:
        df["death_binary"] = 0

    return df

def cluster_countries(country_df, k=4):
    df = country_df.copy()
    df["Mortality_Rate"] = np.where(df["Confirmed"] > 0, df["Deaths"] / df["Confirmed"], 0.0)
    df["Recovery_Rate"] = np.where(df["Confirmed"] > 0, df["Recovered"] / df["Confirmed"], 0.0)

    X = df[["Confirmed", "Deaths", "Recovered", "Mortality_Rate", "Recovery_Rate"]].astype(float).values
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    p = pca.fit_transform(X_scaled)
    df["PCA1"], df["PCA2"] = p[:, 0], p[:, 1]
    return df

def global_latest_totals(covid_df):
    ts = (
        covid_df.groupby("ObservationDate")[["Confirmed", "Deaths", "Recovered"]]
        .sum()
        .reset_index()
        .sort_values("ObservationDate")
    )
    if ts.empty:
        return None
    latest = ts.iloc[-1]
    ratio = (latest["Deaths"] / latest["Recovered"]) if latest["Recovered"] > 0 else np.inf
    return {
        "date": latest["ObservationDate"],
        "confirmed": int(latest["Confirmed"]),
        "deaths": int(latest["Deaths"]),
        "recovered": int(latest["Recovered"]),
        "ratio": ratio,
        "ts": ts,
    }

def highest_two(country_latest):
    ranked = country_latest.sort_values("Confirmed", ascending=False).reset_index(drop=True)
    if ranked.empty:
        return None, None
    top1 = ranked.iloc[0]
    top2 = ranked.iloc[1] if len(ranked) > 1 else None
    return top1, top2

def demographics(line_df):
    # Gender
    gender_counts = line_df["gender"].value_counts(dropna=False)
    # Age stats
    age_stats = line_df["age_numeric"].describe()
    # Age groups
    ag_counts = line_df["age_group"].value_counts().sort_index()
    return gender_counts, age_stats, ag_counts

def mortality_by_age(line_df):
    d = line_df.dropna(subset=["age_group"])
    if d.empty:
        return pd.DataFrame()
    out = d.groupby("age_group").agg(
        deaths=("death_binary", "sum"),
        total=("death_binary", "count")
    ).reset_index()
    out["mortality_rate"] = np.where(out["total"] > 0, 100.0 * out["deaths"] / out["total"], 0.0)
    return out

# ------------------ Load + Clean ------------------
try:
    covid_raw, line_raw = load_data()
except Exception as e:
    st.error(f"Could not load CSVs. Make sure files are next to this script:\n- {COVID_PATH}\n- {LINE_PATH}\n\nError: {e}")
    st.stop()

try:
    covid_df, country_latest = clean_covid(covid_raw)
    line_df = clean_line_list(line_raw)
except Exception as e:
    st.error(f"Cleaning failed: {e}")
    st.stop()

# Precompute clustering (k=4)
cluster_df = cluster_countries(country_latest, k=4)

# ------------------ UI ------------------
st.title("🦠 COVID-19 Data Analysis (Simple)")
st.caption("Auto-loads local CSVs, cleans data, clusters countries, and answers key queries.")

with st.sidebar:
    page = st.radio("Go to", ["Query Results", "Clustering", "Visualizations"], index=0)

# ---------- Page: Query Results ----------
if page == "Query Results":
    st.header("Query Results")

    # i) Highest and second-highest affected areas (by Confirmed)
    st.subheader("i) Highest Affected Areas")
    t1, t2 = highest_two(country_latest)
    col1, col2 = st.columns(2)
    if t1 is not None:
        with col1:
            st.info("Highest Affected")
            st.write(f"Country: {t1['Country/Region']}")
            st.write(f"Confirmed: {int(t1['Confirmed']):,}")
            st.write(f"Deaths: {int(t1['Deaths']):,}")
            st.write(f"Recovered: {int(t1['Recovered']):,}")
    if t2 is not None:
        with col2:
            st.info("Second Highest")
            st.write(f"Country: {t2['Country/Region']}")
            st.write(f"Confirmed: {int(t2['Confirmed']):,}")
            st.write(f"Deaths: {int(t2['Deaths']):,}")
            st.write(f"Recovered: {int(t2['Recovered']):,}")

    # ii) Mortality vs Recovery ratio (latest global totals)
    st.subheader("ii) Mortality vs Recovery Ratio")
    gl = global_latest_totals(covid_df)
    if gl:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Date", gl["date"].date().isoformat())
        with c2: st.metric("Confirmed", f"{gl['confirmed']:,}")
        with c3: st.metric("Deaths", f"{gl['deaths']:,}")
        with c4: st.metric("Recovered", f"{gl['recovered']:,}")
        with c5: st.metric("Deaths/Recovered", f"{gl['ratio']:.4f}" if np.isfinite(gl["ratio"]) else "∞")

    # iii) Demographic tendency
    st.subheader("iii) Demographic Tendency (Age, Gender)")
    g_counts, age_stats, ag_counts = demographics(line_df)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Gender Distribution")
        fig_g = px.pie(
            pd.DataFrame({"Gender": g_counts.index, "Count": g_counts.values}),
            values="Count", names="Gender", title="Gender Distribution"
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with col2:
        st.write("Age Statistics")
        if age_stats["count"] > 0:
            st.write(f"Mean: {age_stats['mean']:.1f}")
            st.write(f"Median: {line_df['age_numeric'].median():.1f}")
            st.write(f"Std: {age_stats['std']:.1f}")
            st.write(f"Range: {age_stats['min']:.1f} - {age_stats['max']:.1f}")
        else:
            st.write("No age data available.")

    if len(ag_counts) > 0:
        st.write("Cases by Age Group")
        fig_ag = px.bar(
            pd.DataFrame({"Age Group": ag_counts.index.astype(str), "Count": ag_counts.values}),
            x="Age Group", y="Count", title="Age Group Distribution", text="Count"
        )
        fig_ag.update_traces(textposition="outside")
        st.plotly_chart(fig_ag, use_container_width=True)

    # iv) Mortality rate among different age groups
    st.subheader("iv) Mortality Rate by Age Group")
    m_age = mortality_by_age(line_df)
    if not m_age.empty:
        fig_m = px.bar(
            m_age.assign(age_group=m_age["age_group"].astype(str)),
            x="age_group", y="mortality_rate",
            title="Mortality Rate by Age Group",
            labels={"age_group": "Age Group", "mortality_rate": "Mortality Rate (%)"},
            text=m_age["mortality_rate"].round(2)
        )
        fig_m.update_traces(textposition="outside")
        st.plotly_chart(fig_m, use_container_width=True)
        st.dataframe(m_age.assign(mortality_rate=m_age["mortality_rate"].round(2)))
    else:
        st.info("Not enough data to compute mortality by age group.")

# ---------- Page: Clustering ----------
elif page == "Clustering":
    st.header("Country Clusters (KMeans, k=4)")
    st.caption("Clustering on Confirmed, Deaths, Recovered, Mortality_Rate, Recovery_Rate")

    # Scatter (PCA)
    fig = px.scatter(
        cluster_df,
        x="PCA1", y="PCA2",
        color=cluster_df["Cluster"].astype(str),
        hover_name="Country/Region",
        hover_data=["Confirmed", "Deaths", "Recovered"],
        title="Clusters (PCA Projection)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Clustered Countries")
    st.dataframe(cluster_df.sort_values(["Cluster", "Confirmed"], ascending=[True, False]).reset_index(drop=True))

# ---------- Page: Visualizations ----------
elif page == "Visualizations":
    st.header("Visualizations")

    # Top 10 countries
    st.subheader("Top 10 Countries (Latest Confirmed)")
    top10 = country_latest.sort_values("Confirmed", ascending=False).head(10)
    fig_top = px.bar(
        top10,
        x="Country/Region", y=["Confirmed", "Deaths", "Recovered"],
        barmode="group",
        title="Top 10 Countries by Cases"
    )
    st.plotly_chart(fig_top, use_container_width=True)

    # Time series
    st.subheader("Global Trends Over Time")
    gl = global_latest_totals(covid_df)
    if gl:
        ts = gl["ts"]
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Confirmed"], mode="lines", name="Confirmed"))
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Deaths"], mode="lines", name="Deaths"))
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Recovered"], mode="lines", name="Recovered"))
        fig_ts.update_layout(title="Global COVID-19 Trends", xaxis_title="Date", yaxis_title="Count", hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)

    # Map
    st.subheader("Geographic Distribution (Latest)")
    fig_map = px.choropleth(
        country_latest,
        locations="Country/Region",
        locationmode="country names",
        color="Confirmed",
        hover_data=["Deaths", "Recovered"],
        color_continuous_scale="Reds",
        title="COVID-19 Cases by Country"
    )
    st.plotly_chart(fig_map, use_container_width=True)
