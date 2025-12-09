"""
Flask Backend for SDG-Aware Movie Recommendation System
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import ast
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ml_models import initialize_models, get_recommendations, classify_sdg

app = Flask(__name__)

# TMDB API key
TMDB_API_KEY = "46ca7da07021f37f46ed0adfdb196195"

# Load data and initialize ML models
def load_data():
    df = pd.read_csv('tmdb_5000_movies.csv')
    
    def extract_genre_names(genre_str):
        try:
            genre_list = ast.literal_eval(genre_str)
            return [genre['name'] for genre in genre_list if 'name' in genre]
        except:
            return []
    
    df['genre_names'] = df['genres'].apply(extract_genre_names)
    return df

df = load_data()
recommender, sdg_classifier = initialize_models(df)

# Pre-compute SDG predictions for all movies
def compute_sdg_predictions():
    predictions = []
    for idx, row in df.iterrows():
        text = row['overview'] if pd.notna(row['overview']) else ''
        preds = classify_sdg(sdg_classifier, text)
        predictions.append(preds)
    return predictions

sdg_predictions = compute_sdg_predictions()
df['ml_sdg_tags'] = [[p[0] for p in preds] for preds in sdg_predictions]
df['ml_sdg_confidence'] = [[p[1] for p in preds] for preds in sdg_predictions]

# SDG categories
SDG_CATEGORIES = [
    {'id': 'sdg4', 'name': 'SDG 4: Quality Education', 'icon': '📚', 'color': '#c5192d'},
    {'id': 'sdg5', 'name': 'SDG 5: Gender Equality', 'icon': '⚧️', 'color': '#ff3a21'},
    {'id': 'sdg10', 'name': 'SDG 10: Reduced Inequalities', 'icon': '⚖️', 'color': '#dd1367'},
    {'id': 'sdg13', 'name': 'SDG 13: Climate Action', 'icon': '🌍', 'color': '#3f7e44'},
    {'id': 'sdg16', 'name': 'SDG 16: Peace & Justice', 'icon': '☮️', 'color': '#00689d'},
]

# Mapping from short ID to full SDG name (used by ML model)
SDG_ID_TO_NAME = {cat['id']: cat['name'] for cat in SDG_CATEGORIES}


def fetch_poster(title):
    """Fetch movie poster from TMDB API."""
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    
    retry_strategy = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    
    try:
        response = http.get(search_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return None


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/sdgs')
def get_sdgs():
    """Get list of SDG categories."""
    return jsonify(SDG_CATEGORIES)


@app.route('/api/movies')
def get_movies():
    """Get movies filtered by SDG category."""
    sdg_id = request.args.get('sdg', '')
    genre = request.args.get('genre', '')
    limit = int(request.args.get('limit', 12))
    
    if not sdg_id:
        return jsonify([])
    
    # Convert short ID to full SDG name used by ML model
    sdg_name = SDG_ID_TO_NAME.get(sdg_id, '')
    if not sdg_name:
        return jsonify([])
    
    # Filter by SDG (using full name)
    filtered = df[df['ml_sdg_tags'].apply(lambda tags: sdg_name in tags)].copy()
    
    # Filter by genre if specified
    if genre:
        filtered = filtered[filtered['genre_names'].apply(lambda g_list: genre in g_list)]
    
    if filtered.empty:
        return jsonify([])
    
    # Sort by confidence
    def get_confidence(row):
        if sdg_name in row['ml_sdg_tags']:
            idx = row['ml_sdg_tags'].index(sdg_name)
            return row['ml_sdg_confidence'][idx] if idx < len(row['ml_sdg_confidence']) else 0
        return 0
    
    filtered['confidence'] = filtered.apply(get_confidence, axis=1)
    filtered = filtered.sort_values('confidence', ascending=False).head(limit)
    
    # Prepare response
    movies = []
    for _, row in filtered.iterrows():
        poster = fetch_poster(row['title'])
        movies.append({
            'title': row['title'],
            'overview': row['overview'] if pd.notna(row['overview']) else '',
            'rating': float(row['vote_average']) if pd.notna(row['vote_average']) else 0,
            'genres': row['genre_names'][:3],
            'sdg_tags': row['ml_sdg_tags'],
            'confidence': round(row['confidence'] * 100, 1),
            'poster': poster
        })
    
    return jsonify(movies)


@app.route('/api/similar/<title>')
def get_similar_movies(title):
    """Get similar movies based on content."""
    limit = int(request.args.get('limit', 10))
    
    similar = get_recommendations(recommender, title, n=limit)
    
    if similar.empty:
        return jsonify({'error': 'Movie not found', 'movies': []})
    
    movies = []
    for _, row in similar.iterrows():
        poster = fetch_poster(row['title'])
        
        # Get SDG tags for this movie
        overview = row['overview'] if pd.notna(row['overview']) else ''
        sdg_preds = classify_sdg(sdg_classifier, overview)
        sdg_tags = [p[0] for p in sdg_preds]
        
        movies.append({
            'title': row['title'],
            'overview': overview,
            'rating': float(row['vote_average']) if pd.notna(row['vote_average']) else 0,
            'genres': row['genre_names'][:3] if 'genre_names' in row else [],
            'similarity': round(row['similarity_score'] * 100, 1),
            'sdg_tags': sdg_tags,
            'poster': poster
        })
    
    return jsonify({'movies': movies, 'query': title})


@app.route('/api/genres')
def get_genres():
    """Get all available genres."""
    all_genres = sorted(set(genre for sublist in df['genre_names'] for genre in sublist))
    return jsonify(all_genres)


@app.route('/api/search')
def search_movies():
    """Search movies by title."""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify([])
    
    matches = df[df['title'].str.lower().str.contains(query.lower(), na=False)].head(limit)
    
    movies = []
    for _, row in matches.iterrows():
        movies.append({
            'title': row['title'],
            'overview': row['overview'] if pd.notna(row['overview']) else '',
            'rating': float(row['vote_average']) if pd.notna(row['vote_average']) else 0,
        })
    
    return jsonify(movies)


if __name__ == '__main__':
    print("🎬 Starting SDG Movie Recommender...")
    print("📊 ML Models loaded successfully!")
    print("🌐 Server running at http://localhost:5001")
    app.run(debug=True, port=5001)
