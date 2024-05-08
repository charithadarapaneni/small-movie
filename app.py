import pickle
import streamlit as st
import numpy as np
from PIL import Image
import base64
import io

st.header("")
model = pickle.load(open('artifacts/model.pkl', 'rb'))
movies_name = pickle.load(open('artifacts/movies_name.pkl','rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
movie_pivot = pickle.load(open('artifacts/movie_pivot.pkl','rb'))

# Increase the font size in the original_title markup
original_title = '<h1 style="font-family: serif; color:white; font-size: 40px;">Movie Recommendation System</h1>'
st.markdown(original_title, unsafe_allow_html=True)

# Set the background image with a semi-transparent overlay
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*QvXq4TCacvs7Z5KNl9V_tQ.jpeg");
    background-size: 100vw 100vh;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

input_style = """
<style>
input[type="text"] {
    background-color: transparent;
    color: #a19eae;
}
div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
</style>
"""
st.markdown(input_style, unsafe_allow_html=True)

def recommend_movies(movie_name):
    movies_list = []
    movie_id = np.where(movie_pivot.index == movie_name)[0][0]
    distance, suggestion = model.kneighbors(movie_pivot.iloc[movie_id,:].values.reshape(1,-1), n_neighbors=6)
    
    for i in range(len(suggestion)):
        movies = movie_pivot.index[suggestion[i]]
        for j in movies:
            movies_list.append(j)
    
    return movies_list

selected_movies = st.selectbox("Type or select a movie", movies_name)

if st.button('Show Recommendation'):
    recommendation_movies = recommend_movies(selected_movies)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommendation_movies[1])
    with col2:
        st.text(recommendation_movies[2])
    with col3:
        st.text(recommendation_movies[3])
    with col4:
        st.text(recommendation_movies[4])
    with col5:
        st.text(recommendation_movies[5])