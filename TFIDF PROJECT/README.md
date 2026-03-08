TF-IDF Document Analysis System
Daniel Ohana (ID: 212889331)

Description
-----------
This project implements a TF-IDF–based document analysis system with an interactive Streamlit interface.

The system allows users to:
- Upload multiple text documents (.txt)
- Apply preprocessing (lowercase, stopword removal, lemmatization)
- Compute TF-IDF representations from scratch
- Extract top keywords per document
- Reduce dimensionality via PCA or SVD
- Visualize document relationships in 2D or 3D space
- Save and reload the TF-IDF model
- Document general and per-document observations

Preprocessing
-------------
All preprocessing is performed inside the TFIDF class based on parameters supplied to __init__():
- Lowercasing (optional)
- Tokenization via regex (\b\w+\b)
- Stopword removal using NLTK (optional)
- Lemmatization using WordNet (optional)

IDF Formula (with smoothing)
----------------------------
The implementation uses smoothed IDF:

idf(w) = log((N + 1) / (df(w) + 1))

where:
- N = total number of documents
- df(w) = number of documents containing term w

Top-N Keywords (configurable)
-----------------------------
The TFIDF class includes a top_n parameter defined in __init__(), controlling how many top keywords are extracted per document.

The Streamlit app calls:
model.get_top_keywords(full_vectors)

Filtered TF-IDF Matrix
----------------------
A compact TF-IDF representation is built using only the union of top keywords across documents.

Dimensionality Reduction
------------------------
Supports PCA (via centered SVD) or direct SVD.  
Visualization is available in both 2D and 3D. The selected document is highlighted in the plots.

Interactive Analysis Interface
------------------------------
Implemented using Streamlit:
- File upload
- Preprocessing options
- TF-IDF model fitting and transformation
- Top-keyword extraction
- Dimensionality reduction and Plotly visualization
- Document selection dropdown
- Per-document and general analysis text areas
- Model saving via pickle

Program Files
-------------
app.py
- Streamlit application
- Handles file uploads, preprocessing options, TF-IDF computation, top-keyword extraction
- Dimensionality reduction (PCA/SVD)
- 2D/3D visualization
- Per-document and general analysis areas
- Model saving via pickle

tfidf.py
- Full TF-IDF implementation from scratch
- Tokenization and preprocessing
- Vocabulary building
- Document frequency and IDF computation
- TF-IDF matrix construction
- fit(), transform(), fit_transform(), get_feature_names()
- get_top_keywords() for top-N keywords per document
- save_to_file() and load_from_file()
- Safe pickling via __getstate__/__setstate__

Installation
------------
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4

Running the Application
-----------------------
streamlit run app.py

Input
-----
Text files in .txt format uploaded through the Streamlit interface.

Output
------
- TF-IDF keyword tables
- 2D or 3D document visualization
- Saved TF-IDF model (.pkl)
- General and per-document analysis
- Filtered TF-IDF keyword matrix

Notes
-----
- The application assumes UTF-8 encoded text files
- Recommended to use at least 5 documents for meaningful visualization
