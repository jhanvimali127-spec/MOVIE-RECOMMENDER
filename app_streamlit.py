import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import ast
import re
import requests
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

.movie-card {
    padding: 10px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 10px;
}

.movie-title {
    font-weight: 600;
    font-size: 16px;
    margin-top: 6px;
}

.small-text {
    color: #777;
    font-size: 13px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("movies_metadata.csv", low_memory=False)

    # Keep only useful columns if they exist
    required = [
        "title", "overview", "genres", "poster_path",
        "release_date", "vote_average", "vote_count",
        "popularity", "runtime", "tagline"
    ]

    for col in required:
        if col not in df.columns:
            df[col] = ""

    df = df[required].copy()

    # Remove missing/duplicate titles
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df = df[df["title"] != ""]
    df = df.drop_duplicates(subset="title").reset_index(drop=True)

    return df


df = load_data()


# ============================================================
# GENRE EXTRACTION
# ============================================================
def extract_genres(value):
    try:
        if pd.isna(value):
            return ""
        data = ast.literal_eval(str(value))

        if isinstance(data, list):
            names = []
            for item in data:
                if isinstance(item, dict) and item.get("name"):
                    names.append(str(item["name"]))
            return " ".join(names)

    except Exception:
        pass

    return str(value) if value else ""


@st.cache_data
def prepare_data(data):
    data = data.copy()

    data["genre_text"] = data["genres"].apply(extract_genres)

    data["overview"] = data["overview"].fillna("").astype(str)
    data["tagline"] = data["tagline"].fillna("").astype(str)

    # Combine important movie information for content-based recommendation
    data["combined_features"] = (
        data["title"].fillna("") + " " +
        data["genre_text"].fillna("") + " " +
        data["overview"].fillna("") + " " +
        data["tagline"].fillna("")
    )

    return data


df = prepare_data(df)


# ============================================================
# TF-IDF MODEL
# ============================================================
@st.cache_resource
def create_model(text_data):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(text_data)

    return vectorizer, matrix


with st.spinner("Loading recommendation model..."):
    vectorizer, tfidf_matrix = create_model(df["combined_features"])


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================
def recommend_movies(movie_title, number_of_movies=12):

    matches = df[
        df["title"].str.lower() == movie_title.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    movie_index = matches.index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    recommendations = []

    for index in similar_indices:
        if index == movie_index:
            continue

        # Recommend only movies that have a usable poster path.
        poster_path = df.loc[index, "poster_path"]
        if pd.isna(poster_path):
            continue

        poster_path = str(poster_path).strip()
        if poster_path == "" or poster_path.lower() in [
            "nan", "none", "null", "0"
        ]:
            continue

        recommendations.append({
            "index": index,
            "score": similarity_scores[index]
        })

        if len(recommendations) >= number_of_movies:
            break

    result = df.loc[
        [item["index"] for item in recommendations]
    ].copy()

    result["similarity_score"] = [
        item["score"] for item in recommendations
    ]

    return result


# ============================================================
# POSTER URL
# ============================================================
# def poster_url(path):
#     if not path or str(path).lower() == "nan":
#         return None

#     path = str(path)

#     if path.startswith("http"):
#         return path

#     return f"https://image.tmdb.org/t/p/w500{path}"


# def poster_url(poster_path):
#     if poster_path is None:
#         return None


#     poster_path = str(poster_path).strip()

#     if poster_path == "" or poster_path.lower() == "nan" or poster_path == "0":
#         return None

#     if not poster_path.startswith("/"):
#         poster_path = "/" + poster_path

#     return f"https://image.tmdb.org/t/p/w500{poster_path}"


# ============================================================
# POSTER URL
# ============================================================

def poster_url(poster_path):
    if poster_path is None:
        return None

    poster_path = str(poster_path).strip()

    if poster_path == "" or poster_path.lower() in [
        "nan", "none", "null", "0"
    ]:
        return None

    if poster_path.startswith("http://") or poster_path.startswith("https://"):
        return poster_path

    if not poster_path.startswith("/"):
        poster_path = "/" + poster_path

    return f"https://image.tmdb.org/t/p/w500{poster_path}"


@st.cache_data(ttl=86400, show_spinner=False)
def load_poster_image(url):
    if not url:
        return None

    try:
        response = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if response.status_code == 200 and response.content:
            content_type = response.headers.get(
                "content-type", ""
            ).lower()

            if content_type.startswith("image/"):
                return response.content

    except requests.RequestException:
        pass

    return None


def show_poster(poster_path):
    url = poster_url(poster_path)

    image_bytes = load_poster_image(url)

    if image_bytes:
        st.image(
            image_bytes,
            use_container_width=True
        )
    else:
        st.markdown(
            """
            <div style="
                height:260px;
                display:flex;
                align-items:center;
                justify-content:center;
                background:#f1f3f5;
                border-radius:10px;
                font-size:42px;
            ">
                🎞️
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MOVIE DETAILS
# ============================================================
def show_movie_details(movie):

    title = movie.get("title", "Unknown Movie")

    st.markdown(f"## 🎬 {title}")

    left, right = st.columns([1, 2])

    # with left:
    #     poster = poster_url(movie.get("poster_path"))

    #     if poster:
    #         st.image(poster, use_container_width=True)
    #     else:
    #         st.info("Poster not available")
    
    with left:
        show_poster(movie.get("poster_path"))

    with right:

        release_date = movie.get("release_date", "")
        rating = movie.get("vote_average", "")
        runtime = movie.get("runtime", "")

        genre = movie.get("genre_text", "")
        overview = movie.get("overview", "")
        tagline = movie.get("tagline", "")

        if tagline and tagline != "nan":
            st.markdown(f"*{tagline}*")

        st.write(f"**Release Date:** {release_date if release_date else 'N/A'}")
        st.write(f"**Rating:** {rating if rating else 'N/A'}")
        st.write(f"**Runtime:** {runtime} minutes" if runtime else "**Runtime:** N/A")
        st.write(f"**Genres:** {genre if genre else 'N/A'}")

        st.markdown("### Overview")

        if overview and overview != "nan":
            st.write(overview)
        else:
            st.write("No overview available.")


# ============================================================
# MOVIE GRID
# ============================================================
def movie_grid(movies, columns=6):

    if movies.empty:
        st.info("No movies found.")
        return

    cols = st.columns(columns)

    for i, (_, movie) in enumerate(movies.iterrows()):
        
        with cols[i % columns]:
            show_poster(movie.get("poster_path"))

        # with cols[i % columns]:

        #     poster = poster_url(movie.get("poster_path"))

        #     if poster:
        #         st.image(poster, use_container_width=True)
        #     else:
        #         st.markdown("🎞️")

            st.markdown(
                f"<div class='movie-title'>{movie['title']}</div>",
                unsafe_allow_html=True
            )

            release = str(movie.get("release_date", ""))[:4]

            if release and release != "nan":
                st.markdown(
                    f"<div class='small-text'>{release}</div>",
                    unsafe_allow_html=True
                )


# ============================================================
# HEADER
# ============================================================
st.markdown(
    "<div class='main-title'>🎬 Movie Recommender</div>",
    unsafe_allow_html=True
)

st.write(
    "Search for a movie and get similar movie recommendations "
    "using a TF-IDF content-based recommendation system."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.header("🎬 Movie Recommender")

    st.write(f"Movies available: **{len(df):,}**")

    number_of_movies = st.slider(
        "Number of recommendations",
        min_value=4,
        max_value=20,
        value=12
    )

    st.divider()

    st.info(
        "Recommendation method:\n\n"
        "TF-IDF + Cosine Similarity"
    )


# ============================================================
# SEARCH
# ============================================================
search_text = st.text_input(
    "🔎 Search Movie",
    placeholder="Example: Toy Story, Batman, Titanic..."
)


if search_text.strip():

    search_lower = search_text.strip().lower()

    # Search titles containing the entered keyword
    search_results = df[
        df["title"].str.lower().str.contains(
            re.escape(search_lower),
            na=False
        )
    ].head(20)

    if search_results.empty:

        st.warning("No movie found. Try another title.")

    else:

        st.subheader("🎯 Search Results")

        selected_title = st.selectbox(
            "Select a movie",
            search_results["title"].tolist()
        )

        selected_movie = df[
            df["title"] == selected_title
        ].iloc[0]

        st.divider()

        show_movie_details(selected_movie)

        st.divider()

        st.subheader("🍿 Recommended Movies")

        recommendations = recommend_movies(
            selected_title,
            number_of_movies
        )

        movie_grid(recommendations, columns=6)

else:

    st.subheader("🎞️ Explore Movies")

    # Show popular movies from the local dataset
    popular_movies = (
        df[
            df["poster_path"].notna()
            & (df["poster_path"].astype(str).str.strip() != "")
            & (~df["poster_path"].astype(str).str.lower().isin(
                ["nan", "none", "null", "0"]
            ))
        ]
        .sort_values(
            by="popularity",
            ascending=False
        )
        .head(12)
    )

    movie_grid(popular_movies, columns=6)


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "Movie Recommendation System | "
    "Content-Based Filtering using TF-IDF and Cosine Similarity"
)
