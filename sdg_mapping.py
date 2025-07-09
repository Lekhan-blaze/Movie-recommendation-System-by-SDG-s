import pandas as pd

# Load the dataset
df = pd.read_csv('tmdb_5000_movies.csv')


sdg_keywords = {
    'SDG 4': ['education', 'learning', 'school', 'teaching', 'student', 'knowledge'],
    'SDG 5': ['gender', 'women', 'equality', 'feminism', 'empowerment'],
    'SDG 10': ['racism', 'poverty', 'discrimination', 'inequality', 'marginalized', 'migration'],
    'SDG 13': ['climate', 'environment', 'pollution', 'sustainability', 'nature'],
    'SDG 16': ['justice', 'peace', 'crime', 'corruption', 'law', 'human rights']
}


def map_sdgs(overview, sdg_keywords):
    matched_sdgs = []
    if isinstance(overview, str):
        overview = overview.lower()
        for sdg, keywords in sdg_keywords.items():
            if any(keyword in overview for keyword in keywords):
                matched_sdgs.append(sdg)
    return matched_sdgs


df['sdg_tags'] = df['overview'].apply(lambda x: map_sdgs(x, sdg_keywords))


print(df[df['sdg_tags'].apply(len) > 0][['title', 'sdg_tags']].head())

# Recommend movies based on selected SDG
def recommend_movies_by_sdg(sdg, df, num_recommendations=5):
    filtered = df[df['sdg_tags'].apply(lambda tags: sdg in tags)]
    return filtered[['title', 'sdg_tags']].head(num_recommendations)

# Example usage:
user_sdg = 'SDG 4'  # You can change this to SDG 4, 5, 10, etc.
recommendations = recommend_movies_by_sdg(user_sdg, df)
print(f"\nMovies related to {user_sdg}:\n")
print(recommendations)