"""
ML Models for SDG-Aware Movie Recommendation System

This module contains:
1. TF-IDF based Content-Based Filtering (Cosine Similarity)
2. SDG Multi-Label Text Classifier
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os

# Paths for model caching
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, 'tfidf_matrix.pkl')
SDG_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'sdg_classifier.pkl')
MLB_PATH = os.path.join(MODEL_DIR, 'mlb.pkl')


class ContentBasedRecommender:
    """TF-IDF based content similarity recommender."""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.df = None
        
    def fit(self, df, text_column='overview'):
        """Fit the TF-IDF vectorizer on movie overviews."""
        self.df = df.copy()
        
        # Handle missing overviews
        self.df['processed_text'] = self.df[text_column].fillna('')
        
        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_text'])
        
        # Cache models
        joblib.dump(self.vectorizer, TFIDF_VECTORIZER_PATH)
        joblib.dump(self.tfidf_matrix, TFIDF_MATRIX_PATH)
        
        return self
    
    def load_cached(self, df):
        """Load cached models if available."""
        if os.path.exists(TFIDF_VECTORIZER_PATH) and os.path.exists(TFIDF_MATRIX_PATH):
            self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            self.tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
            self.df = df.copy()
            return True
        return False
    
    def get_similar_movies(self, movie_title, n=10):
        """Get top N similar movies based on content similarity."""
        if self.df is None or self.tfidf_matrix is None:
            return pd.DataFrame()
        
        # Find the movie index
        matches = self.df[self.df['title'].str.lower() == movie_title.lower()]
        if matches.empty:
            # Try partial match
            matches = self.df[self.df['title'].str.lower().str.contains(movie_title.lower(), na=False)]
        
        if matches.empty:
            return pd.DataFrame()
        
        idx = matches.index[0]
        
        # Get the position in the matrix
        try:
            pos = self.df.index.get_loc(idx)
        except KeyError:
            return pd.DataFrame()
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(self.tfidf_matrix[pos:pos+1], self.tfidf_matrix).flatten()
        
        # Get top similar movies (excluding itself)
        similar_indices = cosine_sim.argsort()[::-1][1:n+1]
        
        # Get similarity scores
        result_df = self.df.iloc[similar_indices].copy()
        result_df['similarity_score'] = cosine_sim[similar_indices]
        
        return result_df


class SDGClassifier:
    """Multi-label SDG classifier using Naive Bayes."""
    
    SDG_DEFINITIONS = {
        'SDG 4: Quality Education': ['education', 'learning', 'school', 'teaching', 'student', 'knowledge', 'teacher', 'university', 'college', 'training'],
        'SDG 5: Gender Equality': ['gender', 'women', 'equality', 'feminism', 'empowerment', 'female', 'sexism', 'women\'s rights', 'girl'],
        'SDG 10: Reduced Inequalities': ['racism', 'poverty', 'discrimination', 'inequality', 'marginalized', 'migration', 'refugee', 'immigrant', 'class', 'segregation'],
        'SDG 16: Peace & Justice': ['justice', 'peace', 'crime', 'corruption', 'law', 'human rights', 'court', 'democracy', 'freedom', 'war']
    }
    
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.mlb = None
        
    def _create_training_labels(self, df, text_column='overview'):
        """Create SDG labels based on keyword matching for training."""
        labels = []
        for text in df[text_column].fillna(''):
            text_lower = text.lower()
            matched_sdgs = []
            for sdg, keywords in self.SDG_DEFINITIONS.items():
                if any(keyword in text_lower for keyword in keywords):
                    matched_sdgs.append(sdg)
            labels.append(matched_sdgs if matched_sdgs else ['None'])
        return labels
    
    def fit(self, df, text_column='overview'):
        """Train the SDG classifier."""
        # Create training data
        texts = df[text_column].fillna('').tolist()
        labels = self._create_training_labels(df, text_column)
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=3000,
            ngram_range=(1, 2)
        )
        X = self.vectorizer.fit_transform(texts)
        
        # Binarize labels
        self.mlb = MultiLabelBinarizer()
        y = self.mlb.fit_transform(labels)
        
        # Train classifier
        self.classifier = OneVsRestClassifier(MultinomialNB())
        self.classifier.fit(X, y)
        
        # Cache models
        joblib.dump(self.vectorizer, SDG_CLASSIFIER_PATH.replace('.pkl', '_vectorizer.pkl'))
        joblib.dump(self.classifier, SDG_CLASSIFIER_PATH)
        joblib.dump(self.mlb, MLB_PATH)
        
        return self
    
    def load_cached(self):
        """Load cached classifier if available."""
        vec_path = SDG_CLASSIFIER_PATH.replace('.pkl', '_vectorizer.pkl')
        if os.path.exists(vec_path) and os.path.exists(SDG_CLASSIFIER_PATH) and os.path.exists(MLB_PATH):
            self.vectorizer = joblib.load(vec_path)
            self.classifier = joblib.load(SDG_CLASSIFIER_PATH)
            self.mlb = joblib.load(MLB_PATH)
            return True
        return False
    
    def predict(self, text, top_n=2, min_confidence=0.1):
        """Predict SDG tags for a given text with confidence scores.
        
        Returns top N SDG tags by probability (above min_confidence threshold).
        """
        if self.classifier is None or self.vectorizer is None:
            return []
        
        if not text or not text.strip():
            return []
        
        X = self.vectorizer.transform([text])
        
        # Get probabilities for all classes
        proba = self.classifier.predict_proba(X)[0]
        
        # Create list of (label, confidence) pairs, excluding 'None'
        sdg_scores = []
        for i, label in enumerate(self.mlb.classes_):
            if label != 'None':
                sdg_scores.append((label, float(proba[i])))
        
        # Sort by confidence descending
        sdg_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N that meet minimum confidence
        result = [(label, conf) for label, conf in sdg_scores[:top_n] if conf >= min_confidence]
        
        return result
    
    def predict_all(self, df, text_column='overview'):
        """Predict SDG tags for all movies in dataframe."""
        predictions = []
        for text in df[text_column].fillna(''):
            preds = self.predict(text)
            predictions.append(preds)
        return predictions


# Convenience functions for the Streamlit app
def initialize_models(df):
    """Initialize and return the ML models."""
    # Content-based recommender
    recommender = ContentBasedRecommender()
    if not recommender.load_cached(df):
        recommender.fit(df)
    else:
        recommender.df = df
    
    # SDG Classifier
    sdg_classifier = SDGClassifier()
    if not sdg_classifier.load_cached():
        sdg_classifier.fit(df)
    
    return recommender, sdg_classifier


def get_recommendations(recommender, movie_title, n=10):
    """Get similar movie recommendations."""
    return recommender.get_similar_movies(movie_title, n)


def classify_sdg(classifier, text):
    """Classify text into SDG categories."""
    return classifier.predict(text)
