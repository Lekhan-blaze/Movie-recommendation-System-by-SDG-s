# 🎬 SDG-Aware Movie Recommendation System

A Machine Learning-powered movie recommendation system that combines **content-based filtering** with **Sustainable Development Goals (SDG) classification**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)

---

## 🧠 How the ML Works

This project uses **two ML models** working together:

### 1️⃣ Content-Based Recommender (TF-IDF + Cosine Similarity)

```
Movie Overview Text → TF-IDF Vectorization → Cosine Similarity → Similar Movies
```

| Step | What Happens |
|------|--------------|
| **TF-IDF Vectorization** | Converts movie descriptions into numerical vectors. Words that are unique to a movie get higher weights, common words get lower weights. |
| **Cosine Similarity** | Measures the angle between two movie vectors. Smaller angle = more similar movies. |
| **Result** | Given a movie, finds other movies with similar plot/themes. |

**Example:** If you search for "The Dark Knight", it finds movies with similar themes (crime, vigilante, hero vs villain).

---

### 2️⃣ SDG Multi-Label Classifier (Naive Bayes)

```
Movie Overview → TF-IDF → One-vs-Rest Naive Bayes → SDG Tags + Confidence
```

| Step | What Happens |
|------|--------------|
| **Training Data Creation** | Movies are labeled with SDGs based on keyword matching (education, climate, equality, etc.) |
| **TF-IDF Vectorization** | Text is converted to numerical features |
| **One-vs-Rest Classifier** | A separate Naive Bayes model is trained for each SDG category |
| **Multi-Label Output** | A movie can have multiple SDG tags (e.g., both "Gender Equality" AND "Peace & Justice") |

**SDG Categories Covered:**
| SDG | Theme | Example Keywords |
|-----|-------|------------------|
| SDG 4 | Quality Education | education, school, learning, teaching |
| SDG 5 | Gender Equality | women, feminism, empowerment, equality |
| SDG 10 | Reduced Inequalities | racism, poverty, discrimination, refugee |
| SDG 16 | Peace & Justice | justice, peace, crime, human rights, war |

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (HTML/CSS/JS)                  │
│  • Dark glassmorphism UI                                     │
│  • SDG category cards                                        │
│  • Movie search & recommendations                            │
└─────────────────────┬───────────────────────────────────────┘
                      │ REST API
┌─────────────────────▼───────────────────────────────────────┐
│                    Flask Backend (app.py)                    │
│  • /api/movies - Get movies by SDG                          │
│  • /api/similar/<title> - Get similar movies                │
│  • /api/search - Search movies                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  ML Models (ml_models.py)                    │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │ ContentBasedRec     │  │ SDGClassifier       │          │
│  │ • TF-IDF Vectorizer │  │ • Naive Bayes       │          │
│  │ • Cosine Similarity │  │ • Multi-Label       │          │
│  └─────────────────────┘  └─────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Data (tmdb_5000_movies.csv)                │
│  • 5000 movies from TMDB                                     │
│  • Title, overview, genres, ratings                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
movie_recc/
├── app.py                      # Flask server & API routes
├── ml_models.py                # ML model classes
├── tmdb_5000_movies.csv        # Movie dataset
├── requirements.txt            # Python dependencies
│
├── 📦 Cached Models (auto-generated)
│   ├── tfidf_vectorizer.pkl    # Content recommender vectorizer
│   ├── tfidf_matrix.pkl        # Pre-computed TF-IDF matrix
│   ├── sdg_classifier.pkl      # Trained SDG classifier
│   ├── sdg_classifier_vectorizer.pkl
│   └── mlb.pkl                 # Multi-label binarizer
│
├── templates/
│   └── index.html              # Main UI template
│
└── static/
    ├── css/style.css           # Dark theme styles
    └── js/app.js               # Frontend logic
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python app.py

# 3. Open in browser
# http://localhost:5000
```

---

## 🔧 Technologies Used

| Category | Technology |
|----------|------------|
| **Backend** | Flask, Python 3.8+ |
| **ML/Data** | scikit-learn, pandas, numpy |
| **Models** | TF-IDF, Cosine Similarity, Naive Bayes |
| **Frontend** | HTML5, CSS3, JavaScript |
| **API** | TMDB (for movie posters) |

---

## 📊 ML Model Details

### TF-IDF Parameters
```python
TfidfVectorizer(
    stop_words='english',     # Remove common words
    max_features=5000,        # Top 5000 words
    ngram_range=(1, 2)        # Single words + bigrams
)
```

### Naive Bayes Classifier
```python
OneVsRestClassifier(MultinomialNB())
# Handles multi-label classification
# Each SDG gets its own binary classifier
```

---

## 👨‍💻 Author

Built for HCAI College Project - SDG Movie Recommendation System

---

## 📜 License

This project is for educational purposes.
