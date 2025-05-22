# Music Recommender System using KNN 

This project implements a music recommendation system using **K-Nearest Neighbors (KNN)** and **unsupervised clustering** techniques on Spotify’s top hits dataset from **2000 to 2019**. The model recommends songs based on acoustic and lyrical features, enabling personalized and accurate suggestions for users. It also uses **cosine similarity**, **text-based fuzzy matching**, and **LIME** for model interpretability.

---

## Dataset

The dataset used is titled **"Top Hits from Spotify (2000–2019)"** and includes various audio and metadata features for over 6000 popular songs across two decades. Features include:
- Danceability, Energy, Loudness, Speechiness, Acousticness
- Instrumentalness, Liveness, Valence, Tempo
- Artist name, Track name, Genre, and more

---

## Features

- Calculates **cosine similarity** between user input and song features for personalized recommendations.
- Uses **K-Means clustering** to group similar songs based on audio characteristics.
- Implements **KNN** for identifying the nearest songs in feature space.
- Uses **FuzzyWuzzy** for accurate text matching on song and artist names.
- Applies **LIME (Local Interpretable Model-Agnostic Explanations)** to explain why a particular recommendation was made.
- Visualizes song clusters using **t-SNE** for dimensionality reduction and analysis.

---

## Technologies Used

- **Python**
- **Pandas, NumPy, Scikit-learn**
- **Cosine Similarity, CountVectorizer**
- **FuzzyWuzzy, LIME**
- **Matplotlib, Seaborn, t-SNE**

---

## How to Run

1. Clone the repository and open the notebook file `music.ipynb`.
2. Ensure the dataset is placed in the appropriate path (e.g., `/input/` directory).
3. Install required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn fuzzywuzzy lime
