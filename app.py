# app.py
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Movie Data Mining & Analytics", layout="wide")

# Step 1: Load and Clean Data
@st.cache_data(show_spinner=False)
def load_data():
    # Ensure files exist in the repo (Streamlit Cloud needs them at app root unless using URLs)
    required = ["Movies.csv", "Ratings.csv", "Users.csv"]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files: {', '.join(missing)}. "
            f"Place them in the repo root or update the path."
        )

    # Use encoding_errors instead of errors (works on pandas 2.x)
    movies = pd.read_csv("Movies.csv", encoding="latin-1", encoding_errors="ignore")
    ratings = pd.read_csv("Ratings.csv", encoding="latin-1", encoding_errors="ignore")
    users = pd.read_csv("Users.csv", encoding="latin-1", encoding_errors="ignore")

    # Extract year from title
    movies["Year"] = pd.to_numeric(
        movies["Title"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
    )

    # Ensure Category exists, then explode multi-category strings
    movies["Category"] = movies["Category"].astype(str)
    movies = movies.assign(Category=movies["Category"].str.split("|")).explode("Category")

    # Merge datasets
    df = ratings.merge(movies, on="MovieID", how="inner")
    dfu = df.merge(users, on="UserID", how="inner")

    # Age mapping for readability
    age_map = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+",
    }
    dfu["AgeGroup"] = dfu["Age"].map(age_map)

    return movies, ratings, users, df, dfu


def compute_all(movies, ratings, users, df, dfu):
    # i) Movies released per year
    movies_per_year = (
        movies.dropna(subset=["Year"])
        .groupby("Year", as_index=False)["MovieID"]
        .nunique()
        .rename(columns={"MovieID": "MovieCount"})
        .sort_values("Year")
    )

    # ii) Highest-rated category each year
    category_rating = df.groupby(["Year", "Category"], as_index=False)["Rating"].mean()
    highest_each_year = category_rating.loc[
        category_rating.groupby("Year")["Rating"].idxmax()
    ]

    # iii) Category + Age group preferences
    age_category_counts = (
        dfu.groupby(["AgeGroup", "Category"], as_index=False)["Rating"]
        .count()
        .rename(columns={"Rating": "Count"})
    )
    preferences = age_category_counts.loc[
        age_category_counts.groupby("AgeGroup")["Count"].idxmax()
    ]

    # iv) Clustering: AgeGroup vs Category
    pivot_age_cat = age_category_counts.pivot_table(
        index="AgeGroup", columns="Category", values="Count", fill_value=0
    )
    scaler_age = StandardScaler()
    X_age = scaler_age.fit_transform(pivot_age_cat)
    n_clusters_age = min(3, len(pivot_age_cat)) if len(pivot_age_cat) > 0 else 1
    kmeans_age = KMeans(n_clusters=n_clusters_age, random_state=42, n_init=10)
    pivot_age_cat = pivot_age_cat.copy()
    pivot_age_cat["Cluster"] = kmeans_age.fit_predict(X_age) if len(pivot_age_cat) else []

    # vi) Year & Category counts
    year_cat_count = (
        movies.dropna(subset=["Year"])
        .groupby(["Year", "Category"], as_index=False)["MovieID"]
        .nunique()
        .rename(columns={"MovieID": "MovieCount"})
        .sort_values(["Year", "Category"])
    )

    # vii) Occupation clusters
    occupation_category = (
        dfu.groupby(["Occupation", "Category"], as_index=False)["Rating"].count()
    )
    pivot_occ_cat = occupation_category.pivot_table(
        index="Occupation", columns="Category", values="Rating", fill_value=0
    )
    scaler_occ = StandardScaler()
    X_occ = scaler_occ.fit_transform(pivot_occ_cat) if len(pivot_occ_cat) else []
    n_clusters_occ = min(5, len(pivot_occ_cat)) if len(pivot_occ_cat) > 0 else 1
    kmeans_occ = KMeans(n_clusters=n_clusters_occ, random_state=42, n_init=10)
    pivot_occ_cat = pivot_occ_cat.copy()
    pivot_occ_cat["Cluster"] = (
        kmeans_occ.fit_predict(X_occ) if len(pivot_occ_cat) else []
    )

    # viii) Occupation + Age clusters
    occ_age_cat = (
        dfu.groupby(["AgeGroup", "Occupation", "Category"], as_index=False)["Rating"]
        .count()
        .rename(columns={"Rating": "Count"})
    )
    pivot_occ_age = occ_age_cat.pivot_table(
        index=["AgeGroup", "Occupation"], columns="Category", values="Count", fill_value=0
    )
    scaler_occ_age = StandardScaler()
    X_occ_age = scaler_occ_age.fit_transform(pivot_occ_age) if len(pivot_occ_age) else []
    n_clusters_occ_age = min(6, len(pivot_occ_age)) if len(pivot_occ_age) > 0 else 1
    kmeans_occ_age = KMeans(n_clusters=n_clusters_occ_age, random_state=42, n_init=10)
    pivot_occ_age = pivot_occ_age.copy()
    pivot_occ_age["Cluster"] = (
        kmeans_occ_age.fit_predict(X_occ_age) if len(pivot_occ_age) else []
    )

    # ix) Reverse: Category → top AgeGroup & Occupation
    category_user_pref = (
        dfu.groupby(["Category", "AgeGroup", "Occupation"], as_index=False)["Rating"]
        .count()
        .rename(columns={"Rating": "Count"})
    )
    top_user_segments = category_user_pref.loc[
        category_user_pref.groupby("Category")["Count"].idxmax()
    ]

    return (
        movies_per_year,
        highest_each_year,
        preferences,
        pivot_age_cat,
        year_cat_count,
        pivot_occ_cat,
        pivot_occ_age,
        top_user_segments,
    )


# Main UI
st.title("Movie Data Mining & Analytics System")

try:
    movies, ratings, users, df, dfu = load_data()
    (
        movies_per_year,
        highest_each_year,
        preferences,
        pivot_age_cat,
        year_cat_count,
        pivot_occ_cat,
        pivot_occ_age,
        top_user_segments,
    ) = compute_all(movies, ratings, users, df, dfu)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    # Show error details in the app while you debug (you can remove later)
    st.exception(e)
    st.stop()

menu = st.sidebar.radio(
    "Choose Analysis",
    [
        "Total number of movies released in each year",
        "Top Category per Year",
        "Preferences by Age Group",
        "Age Group Clusters",
        "Year & Category Counts",
        "Occupation Clusters",
        "Occupation+Age Clusters",
        "Category -> User Segments",
    ],
)

if menu == "Total number of movies released in each year":
    st.subheader("Total number of movies released in each year")
    st.dataframe(movies_per_year, use_container_width=True)
    if not movies_per_year.empty:
        st.line_chart(movies_per_year.set_index("Year"))

elif menu == "Top Category per Year":
    st.subheader("Highest Rated Category in Each Year")
    st.dataframe(highest_each_year, use_container_width=True)

elif menu == "Preferences by Age Group":
    st.subheader("Most Liked Categories by Age Group")
    st.dataframe(preferences, use_container_width=True)

elif menu == "Age Group Clusters":
    st.subheader("Clustering Age Groups vs Categories")
    st.dataframe(pivot_age_cat, use_container_width=True)

elif menu == "Year & Category Counts":
    st.subheader("Movies Released Each Year by Category")
    st.dataframe(year_cat_count, use_container_width=True)

elif menu == "Occupation Clusters":
    st.subheader("Clustering of Occupations vs Categories")
    st.dataframe(pivot_occ_cat, use_container_width=True)

elif menu == "Occupation+Age Clusters":
    st.subheader("Clustering of Occupation + AgeGroups vs Categories")
    st.dataframe(pivot_occ_age, use_container_width=True)

elif menu == "Category -> User Segments":
    st.subheader("Most Likely AgeGroup & Occupation per Category")
    st.dataframe(top_user_segments, use_container_width=True)
