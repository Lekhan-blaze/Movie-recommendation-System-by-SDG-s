# SDG-Aware Movie Recommendation System
## Project Explanation for Presentation

---

## 🎯 What Does This Project Do?

This is a **Machine Learning-powered movie recommendation system** that:
1. **Recommends similar movies** based on plot/content similarity
2. **Classifies movies by UN Sustainable Development Goals (SDGs)** - helping users find movies about education, equality, justice, etc.

---

## 🧠 Machine Learning Concepts Used

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

**What is it?**
TF-IDF is a technique to convert text into numbers that computers can understand.

**How it works:**
- **Term Frequency (TF)**: How often a word appears in a movie description
- **Inverse Document Frequency (IDF)**: How rare that word is across ALL movies

**Example:**
- The word "the" appears in almost every movie → LOW TF-IDF score
- The word "heist" only appears in crime movies → HIGH TF-IDF score

**Formula:**
```
TF-IDF = TF × log(Total Documents / Documents containing the word)
```

**Why we use it:**
It finds the most **important/unique words** in each movie description, ignoring common words.

---

### 2. Cosine Similarity

**What is it?**
A way to measure how "similar" two movies are based on their TF-IDF vectors.

**How it works:**
- Each movie becomes a vector (list of numbers) after TF-IDF
- Cosine similarity measures the **angle** between two vectors
- Smaller angle = More similar movies

**Formula:**
```
Cosine Similarity = (A · B) / (||A|| × ||B||)
```

**Visual Example:**
```
Movie A: [0.5, 0.8, 0.1]  (action, heist, romance)
Movie B: [0.6, 0.7, 0.2]  (action, heist, romance)
→ High cosine similarity (similar movies!)

Movie C: [0.1, 0.1, 0.9]  (romance-heavy)
→ Low cosine similarity with A and B (different type)
```

---

### 3. Naive Bayes Classifier

**What is it?**
A probabilistic machine learning algorithm for classification.

**How it works:**
- Based on **Bayes' Theorem** from probability
- Assumes features (words) are independent of each other (that's the "naive" part)
- Calculates: "Given these words, what's the probability this movie is about Peace & Justice?"

**Formula (Bayes' Theorem):**
```
P(SDG | Words) = P(Words | SDG) × P(SDG) / P(Words)
```

**In plain English:**
"What's the probability this movie is about Peace & Justice, given it contains words like 'justice', 'crime', 'war'?"

---

### 4. Multi-Label Classification (One-vs-Rest)

**What is it?**
A technique to handle cases where one item can belong to **multiple categories**.

**Why we need it:**
A movie can be about BOTH "Gender Equality" AND "Reduced Inequalities" at the same time!

**How it works:**
- Train a **separate binary classifier** for each SDG
- Each classifier answers: "Is this movie about THIS specific SDG? Yes/No"
- Combine all results to get multiple labels

```
Movie: "A woman fights for equal rights in a poor community"
├── SDG 5 Classifier → YES (Gender Equality)
├── SDG 10 Classifier → YES (Reduced Inequalities)
├── SDG 4 Classifier → NO
└── SDG 16 Classifier → NO

Result: [SDG 5, SDG 10]
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│  (HTML/CSS/JavaScript - Dark Theme with Glassmorphism)  │
└────────────────────────┬────────────────────────────────┘
                         │ REST API Calls
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   FLASK BACKEND                          │
│  • Handles HTTP requests                                │
│  • Routes: /api/movies, /api/similar, /api/sdgs         │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌───────────────────┐           ┌───────────────────┐
│ Content-Based     │           │ SDG Classifier    │
│ Recommender       │           │                   │
│                   │           │ • Naive Bayes     │
│ • TF-IDF          │           │ • Multi-Label     │
│ • Cosine Sim      │           │ • One-vs-Rest     │
└───────────────────┘           └───────────────────┘
        │                                 │
        └────────────────┬────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              TMDB 5000 MOVIES DATASET                    │
│  • 5000 movies with titles, descriptions, ratings       │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 SDG Categories Covered

| SDG | Goal | Example Keywords |
|-----|------|------------------|
| SDG 4 | Quality Education | education, school, learning, teaching, student |
| SDG 5 | Gender Equality | women, feminism, empowerment, equality |
| SDG 10 | Reduced Inequalities | racism, poverty, discrimination, refugee |
| SDG 16 | Peace & Justice | justice, peace, crime, human rights, war |

---

## 🔄 How the Recommendation Works

### Step-by-Step Process:

**1. User searches for "Inception"**
```
User Input: "Inception"
```

**2. System finds Inception in database**
```
Movie Found: "Inception"
Overview: "A thief who steals corporate secrets through dream-sharing technology..."
```

**3. TF-IDF converts text to numbers**
```
Inception Vector: [0.3, 0.0, 0.5, 0.8, 0.1, ...]
                   │         │    │
                   dreams    heist technology
```

**4. Cosine similarity finds similar movies**
```
Compare with all 5000 movies:
├── "The Dark Knight" → 85% similar
├── "Interstellar" → 82% similar  
├── "The Prestige" → 78% similar
└── ...
```

**5. SDG Classifier tags each result**
```
"The Dark Knight" → [SDG 16: Peace & Justice]
"Interstellar" → [SDG 4: Quality Education]
```

**6. Return results to user**

---

## 🛠️ Technologies Used

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | HTML, CSS, JavaScript | User interface |
| Backend | Flask (Python) | API server |
| ML Library | scikit-learn | TF-IDF, Naive Bayes, Cosine Similarity |
| Data Processing | pandas, numpy | Data manipulation |
| API | TMDB API | Movie poster images |

---

## 📁 Key Files Explained

| File | Purpose |
|------|---------|
| `app.py` | Flask server, API endpoints, main application logic |
| `ml_models.py` | ML classes (ContentBasedRecommender, SDGClassifier) |
| `tmdb_5000_movies.csv` | Dataset with 5000 movies |
| `*.pkl` files | Cached/trained ML models (loaded on startup) |

---

## 💡 Key Talking Points for Presentation

1. **"This uses TF-IDF to understand movie descriptions"**
   - Converts text to numbers, gives higher weight to unique words

2. **"Cosine similarity measures how alike two movies are"**
   - Like measuring the angle between two arrows

3. **"Naive Bayes predicts which SDG goals each movie aligns with"**
   - Uses probability to classify movies

4. **"One-vs-Rest allows a movie to have multiple SDG tags"**
   - Each SDG has its own binary classifier

5. **"The system combines content similarity with social impact classification"**
   - Unique combination of recommendation + SDG awareness

---

## ❓ Common Questions & Answers

**Q: Why TF-IDF instead of simple word count?**
A: TF-IDF ignores common words like "the" and "a", focusing on meaningful words.

**Q: Why Naive Bayes?**
A: It's simple, fast, works well with text, and gives probability scores.

**Q: Why is it called "Naive"?**
A: It naively assumes all words are independent (which isn't true, but works well in practice).

**Q: How accurate is the SDG classification?**
A: The model shows confidence scores (0-100%) for each prediction.

**Q: What dataset is used?**
A: TMDB 5000 Movies dataset - real movies with descriptions, ratings, and genres.

---

## 🎓 Learning Outcomes Demonstrated

1. **Text Processing** - TF-IDF vectorization
2. **Machine Learning** - Supervised classification (Naive Bayes)
3. **Similarity Measures** - Cosine similarity for recommendations
4. **Multi-label Classification** - One-vs-Rest strategy
5. **Web Development** - Full-stack Flask application
6. **API Integration** - TMDB API for movie posters

---

*This project demonstrates practical application of ML algorithms for both content-based recommendation and text classification, while addressing social awareness through UN SDG goals.*
