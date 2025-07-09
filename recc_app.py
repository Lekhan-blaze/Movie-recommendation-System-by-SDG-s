import pandas as pd
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(layout="wide")

#  API key 
TMDB_API_KEY = "46ca7da07021f37f46ed0adfdb196195"

def fetch_poster(title):
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title
    }

    # Set up retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)

    try:
        response = http.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster: {e}")
    
    return None

# Load data
df = pd.read_csv('tmdb_5000_movies.csv')

import ast  # Needed to safely evaluate stringified lists  # noqa: E402

def extract_genre_names(genre_str):
    try:
        genre_list = ast.literal_eval(genre_str)
        return [genre['name'] for genre in genre_list if 'name' in genre]
    except:  # noqa: E722
        return []

df['genre_names'] = df['genres'].apply(extract_genre_names)

# SDG keywords
sdg_keywords = {
    'SDG 4: Quality Education': ['education', 'learning', 'school', 'teaching', 'student', 'knowledge'],
    'SDG 5: Gender Equality': ['gender', 'women', 'equality', 'feminism', 'empowerment'],
    'SDG 10: Reduced Inequalities': ['racism', 'poverty', 'discrimination', 'inequality', 'marginalized', 'migration'],
    'SDG 13: Climate Action': ['climate', 'environment', 'pollution', 'sustainability', 'nature'],
    'SDG 16: Peace & Justice': ['justice', 'peace', 'crime', 'corruption', 'law', 'human rights']
}

# SDG Mapping Function
def map_sdgs(overview):
    matched_sdgs = []
    if isinstance(overview, str):
        overview = overview.lower()
        for sdg, keywords in sdg_keywords.items():
            if any(keyword in overview for keyword in keywords):
                matched_sdgs.append(sdg)
    return matched_sdgs

# Apply SDG tags
df['sdg_tags'] = df['overview'].apply(map_sdgs)

# UI
st.title("🎬 SDG-Aware Movie Recommendation System")


# Search bar
search_query = st.text_input("🔍 Search for a movie by title")

# Search logic
if search_query:
    searched = df[df['title'].str.contains(search_query, case=False, na=False)]
    st.subheader("🔎 Search Results")
    if not searched.empty:
        for _, row in searched.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    poster_url = fetch_poster(row['title'])
                    if poster_url:
                        st.image(poster_url, width=150)
                with col2:
                    st.markdown(f"### 🎬 {row['title']}")
                    st.write("**SDG Tags:**", ", ".join(row['sdg_tags']) if row['sdg_tags'] else "None")
                    st.write("**Genres:**", ", ".join(row['genre_names']) if row['genre_names'] else "Unknown")
                    st.write("**Rating:**", row.get('vote_average', 'N/A'))
                st.markdown("---")
    else:
        st.warning("No movies found for your search.")

# Dropdown for SDG selection
sdg_options = list(sdg_keywords.keys())
selected_sdg = st.selectbox(
    label="🎯 Select SDG",
    options=[""] + sdg_options,
    format_func=lambda x: "" if x == "" else x
)

# Get unique genres
all_genres = sorted(set(genre for sublist in df['genre_names'] for genre in sublist))

# Genre multiselect
selected_genres = st.multiselect("🎭 Filter by Genre:", all_genres)
        
# Number of movies
num_recs = st.slider("Number of recommendations:", min_value=1, max_value=20, value=10)

# Filter based on SDG
# filtered = df[df['sdg_tags'].apply(lambda tags: selected_sdg in tags)]
# Conditional filtering logic
if selected_sdg != " ":  # Only run if SDG is selected
    if search_query:
        filtered = df[df['title'].str.contains(search_query, case=False, na=False)]
        st.subheader(f"🔎 Search Results for '{search_query}'")
    else:
        filtered = df[df['sdg_tags'].apply(lambda tags: selected_sdg in tags)]
        if selected_genres:
            filtered = filtered[filtered['genre_names'].apply(lambda g_list: any(g in g_list for g in selected_genres))]
        st.subheader(f"🎯 Top {num_recs} Recommendations for {selected_sdg}")

    # Display results
    movie_chunks = [filtered.head(num_recs).iloc[i:i+5] for i in range(0, num_recs, 5)]

    for chunk in movie_chunks:
        cols = st.columns(5)
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                with st.container():
                    poster_url = fetch_poster(row['title'])
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.markdown("🎞️ *No poster available*")

                    st.markdown(f"**🎬 {row['title']}**", help=row['overview'])
                    st.markdown(f"**🎯 SDGs:** {', '.join(row['sdg_tags']) if row['sdg_tags'] else 'None'}")
                    st.markdown(f"**🎭 Genres:** {', '.join(row['genre_names']) if row['genre_names'] else 'Unknown'}")
                    st.markdown(f"**⭐ Rating:** {row.get('vote_average', 'N/A')}")
                    # Feedback buttons
                    unique_key = f"{row.name}_{row.get('title', 'movie')}"

                    col_like, col_dislike = st.columns([1, 1])

                    with col_like:
                         if st.button("👍 Like", key=f"like_{unique_key}"):
                            st.success("You liked this movie!")

                    with col_dislike:
                        if st.button("👎 Dislike", key=f"dislike_{unique_key}"):
                            st.warning("You disliked this movie!")                   
    
