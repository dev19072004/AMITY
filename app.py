#importing the libraries
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier

import streamlit as st

# Step 1: Load and Clean Data
@st.cache_data
def load_data():
    # Load datasets
    movies = pd.read_csv("Movies.csv",encoding='latin-1')
    ratings = pd.read_csv("Ratings.csv",encoding='latin-1')
    users = pd.read_csv("Users.csv",encoding='latin-1')

    movies['Year'] = movies['Title'].str.extract(r'(\d{4})')
    movies['Year'] = pd.to_numeric(movies['Year'], errors='coerce')
    movies = movies.assign(Category=movies['Category'].str.split('|')).explode('Category')
    df = ratings.merge(movies, on="MovieID")
    dfu = df.merge(users, on="UserID")

    # Age mapping for readability
    age_map = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+"
    }
    dfu['AgeGroup'] = dfu['Age'].map(age_map)

    return movies, ratings, users, df, dfu

movies, ratings, users, df, dfu = load_data()

# Step 2: Queries

# i) Movies released per year
movies_per_year = movies.groupby('Year')['MovieID'].nunique().reset_index(name='MovieCount')

# ii) Highest-rated category each year
category_rating = df.groupby(['Year','Category'])['Rating'].mean().reset_index()
highest_each_year = category_rating.loc[category_rating.groupby('Year')['Rating'].idxmax()]

# iii) Category + Age group preferences
age_category_counts = dfu.groupby(['AgeGroup','Category'])['Rating'].count().reset_index(name='Count')
preferences = age_category_counts.loc[age_category_counts.groupby('AgeGroup')['Count'].idxmax()]

# iv) Clustering: AgeGroup vs Category
pivot_age_cat = age_category_counts.pivot(index='AgeGroup', columns='Category', values='Count').fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_age_cat)
kmeans_age = KMeans(n_clusters=3, random_state=42, n_init=10)
pivot_age_cat['Cluster'] = kmeans_age.fit_predict(X_scaled)

# vi) Year & Category counts
year_cat_count = movies.groupby(['Year','Category'])['MovieID'].nunique().reset_index(name='MovieCount')

# vii) Occupation clusters
occupation_category = dfu.groupby(['Occupation','Category'])['Rating'].count().reset_index()
pivot_occ_cat = occupation_category.pivot(index='Occupation', columns='Category', values='Rating').fillna(0)
X_occ = scaler.fit_transform(pivot_occ_cat)
kmeans_occ = KMeans(n_clusters=5, random_state=42, n_init=10)
pivot_occ_cat['Cluster'] = kmeans_occ.fit_predict(X_occ)

# viii) Occupation + Age clusters
occ_age_cat = dfu.groupby(['AgeGroup','Occupation','Category'])['Rating'].count().reset_index(name='Count')
pivot_occ_age = occ_age_cat.pivot_table(index=['AgeGroup','Occupation'], columns='Category', values='Count').fillna(0)
X_occ_age = scaler.fit_transform(pivot_occ_age)
kmeans_occ_age = KMeans(n_clusters=6, random_state=42, n_init=10)
pivot_occ_age['Cluster'] = kmeans_occ_age.fit_predict(X_occ_age)

# ix) Reverse: Category → top AgeGroup & Occupation
category_user_pref = dfu.groupby(['Category','AgeGroup','Occupation'])['Rating'].count().reset_index(name='Count')
top_user_segments = category_user_pref.loc[category_user_pref.groupby('Category')['Count'].idxmax()]

# Step 3: Streamlit Interface
st.title("Movie Data Mining & Analytics System")

menu = st.sidebar.radio("Choose Analysis", [
    "Total number of movies released in each year",
    "Top Category per Year",
    "Preferences by Age Group",
    "Age Group Clusters",
    "Year & Category Counts",
    "Occupation Clusters",
    "Occupation+Age Clusters",
    "Category -> User Segments"
])

if menu == "Total number of movies released in each year":
    st.subheader("Total number of movies released in each year")
    st.dataframe(movies_per_year)
    if not movies_per_year.empty:
        st.line_chart(movies_per_year.set_index('Year'))

elif menu == "Top Category per Year":
    st.subheader("Highest Rated Category in Each Year")
    st.dataframe(highest_each_year)

elif menu == "Preferences by Age Group":
    st.subheader("Most Liked Categories by Age Group")
    st.dataframe(preferences)

elif menu == "Age Group Clusters":
    st.subheader("Clustering Age Groups vs Categories")
    st.dataframe(pivot_age_cat)

elif menu == "Year & Category Counts":
    st.subheader("Movies Released Each Year by Category")
    st.dataframe(year_cat_count)

elif menu == "Occupation Clusters":
    st.subheader("Clustering of Occupations vs Categories")
    st.dataframe(pivot_occ_cat)

elif menu == "Occupation+Age Clusters":
    st.subheader("Clustering of Occupation + AgeGroups vs Categories")
    st.dataframe(pivot_occ_age)

elif menu == "Category -> User Segments":
    st.subheader("Most Likely AgeGroup & Occupation per Category")
    st.dataframe(top_user_segments)
