import numpy as np
import pandas as pd
from IPython.core.display import HTML
import streamlit as st
import requests
import os

st.set_page_config(layout="wide")
st.title("Movie Recommendation System")

#Rmat = pd.read_csv('Rmat.csv')
ms_link = "https://www.dropbox.com/scl/fi/vnr7me48bsll7napndi8y/movie_similarity.csv?rlkey=t35qmwahba6bus48j0a1oaief&st=2ajronvj&dl=1"
ms_filename = "ms.csv"
response = requests.get(ms_link)
with open(ms_filename, "wb") as file:
    file.write(response.content)
S = pd.read_csv(ms_filename, index_col=0)
os.remove(ms_filename)

Rmat_link = "https://www.dropbox.com/scl/fi/qp5gh27wnifszx53ohwhn/Rmat.csv?rlkey=o84nshijan8lsf1a09jnkxrvd&st=jeyna1l1&dl=1"
Rm_filename = "Rm.csv"
response = requests.get(Rmat_link)
with open(Rm_filename, "wb") as file:
    file.write(response.content)
Rmat = pd.read_csv(Rm_filename)
os.remove(Rm_filename)

mov_link = "https://www.dropbox.com/scl/fi/2a9qtxv80h9mmb805ofni/movies.dat?rlkey=6d6rzzjulzzjfoec5jpg9pikf&st=qefs4tsd&dl=1"
movies_filename = "mov.dat"
response = requests.get(mov_link)
with open(movies_filename, "wb") as file:
    file.write(response.content)
movies = pd.read_csv(movies_filename, sep='::', engine = 'python',
                     encoding="ISO-8859-1", header = None)
movies.columns = ['MovieID', 'Title', 'Genres']
os.remove(movies_filename) 

cleaned_columns = Rmat.columns.str[1:]
filtered_movies = movies[movies['MovieID'].isin(cleaned_columns.astype(int))]
filtered_movies.reset_index(drop=True, inplace=True)

ratings_counts = Rmat.notnull().sum().sort_values(ascending=False)

small_image_url = "https://liangfgithub.github.io/MovieImages/"

top_100_movie_ids = ratings_counts.index[0:100]
top_100_movie_ids = top_100_movie_ids.str[1:].astype(int)
top_100_movies = filtered_movies[filtered_movies['MovieID'].isin(top_100_movie_ids)]
top_100_movies = top_100_movies.set_index('MovieID').loc[top_100_movie_ids].reset_index()
top_100_movies = top_100_movies.drop('Genres', axis=1)
top_100_movies.rename(columns={'index': 'MovieID'}, inplace=True)
top_100_movies['Image'] = '<img src="' + small_image_url + top_100_movies['MovieID'].astype(str) + '.jpg?raw=true"></img>'

#S = pd.read_csv("movie_similarity.csv", index_col=0)

def keep_top_30_similarities(row):
    ranked = row.rank(method='min', ascending=False)
    top_30 = ranked <= 30
    return row.where(top_30)

S = S.apply(keep_top_30_similarities, axis=1)

def myIBCF(newuser):
    predictions = np.full(newuser.shape[0], np.nan)

    if newuser.isna().all():
        top_10_movies = top_100_movies.head(10)
        return top_10_movies['MovieID']
    
    for i in range(len(newuser)):
        if not np.isnan(newuser.iloc[i]):
            continue

        similar_movies = S.iloc[i, :].copy()
        rated_indices = ~newuser.isna()
        S_i = similar_movies[rated_indices].to_numpy()
        W_j = newuser[rated_indices].to_numpy()

        numerator = np.nansum(S_i * W_j)
        denominator = np.nansum(S_i)

        if denominator > 0:
            predictions[i] = numerator / denominator

    predictions_df = pd.DataFrame({
        "MovieID": Rmat.columns,
        "Prediction": predictions
    }).sort_values(by="Prediction", ascending=False)

    top_10 = predictions_df.dropna().head(10)

    if len(top_10) < 10:
        movie_popularity = Rmat.notna().sum().sort_values(ascending=False)
        popular_movies = movie_popularity.index.difference(top_10["MovieID"])
        additional_movies = movie_popularity.loc[popular_movies].head(10 - len(top_10)).index
        top_10 = pd.concat([top_10, pd.DataFrame({"MovieID": additional_movies, "Prediction": [None] * len(additional_movies)})])
    
    top_10['MovieID'] = top_10['MovieID'].str[1:].astype(int)
    # Step 5: Format the output as required
    #return [str(movie) for movie in top_10["MovieID"]]
    return top_10.reset_index(drop=True)


if "ratings" not in st.session_state:
    st.session_state.ratings = {movie_id: None for movie_id in filtered_movies['MovieID']}  # Default is None

# Display movies with a 5-radio rating system
with st.expander("The 100 Most Popular Movies(rate as many movies as you can)"):
    for i in range(0, len(top_100_movies), 5):
        row_data = top_100_movies.iloc[i:i + 5]
        cols = st.columns(5)  # Create 5 fixed columns in each row
        for col, (_, row) in zip(cols, row_data.iterrows()):
            with col:
                # Display movie image and title
                st.image(small_image_url + str(row['MovieID']) + '.jpg?raw=true', width=100)
                st.text(row['Title'])

                # Add radio button for rating with no default selection
                rating = st.radio(
                    label="",  # No label
                    options=[1, 2, 3, 4, 5],  # Options for rating
                    index=None if st.session_state.ratings[row['MovieID']] is None else st.session_state.ratings[row['MovieID']] - 1,
                    key=f"rating_{row['MovieID']}",
                    horizontal=True
                )

                st.session_state.ratings[row['MovieID']] = rating

# Display submitted ratings
#st.write("Your Ratings:")
ratings_series = pd.Series(st.session_state.ratings)
#st.write(ratings_series)
#st.write(ratings_series.shape)
#st.write(ratings_series[0])

with st.expander('Recommendations tailored for you'):
    if st.button('Get Recommendations'):
        newuser = ratings_series
        newuser.index = newuser.index.map(lambda x: 'm' + str(x))
        top_10 = myIBCF(newuser)
        #top_10['MovieID'] = top_10['MovieID'].str[1:].astype(int)
        df_top_10 = pd.merge(top_10, filtered_movies, on='MovieID', how='left')
        #st.write(top_10)

        for i in range(0, len(df_top_10), 5):
            row_data = df_top_10.iloc[i:i + 5]
            cols = st.columns(5)  # Create 5 fixed columns in each row
            for col, (_, row) in zip(cols, row_data.iterrows()):
                with col:
                    st.image(small_image_url + str(row['MovieID']) + '.jpg?raw=true', width=100)
                    st.text(row['Title'])