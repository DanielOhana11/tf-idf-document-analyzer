# tfidf.py
import re
import string
import pickle
from typing import List
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TFIDF:
    """
    TF-IDF implementation with internal preprocessing.

    Preprocessing:
        1. Lowercase (optional)
        2. Tokenization via regex r'\b\w+\b'
        3. Stopword removal (optional)
        4. Lemmatization (optional)

    IDF:
        idf(w) = log((N + 1) / (df(w) + 1))
    """
    
    def __init__(self,
                 lowercase: bool = True,
                 remove_stopwords: bool = False,
                 lemmatize: bool = False,
                 top_n: int = 10):
        """
        Initialize the TFIDF object with preprocessing options.

        Parameters
        ----------
        lowercase : bool, default=True
        remove_stopwords : bool, default=False
        lemmatize : bool, default=False
        top_n : int, default=10
            Number of top keywords to extract per document (used for visualization).
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        # validate and store top_n
        self.top_n = int(top_n)
        if self.top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

        # Learned parameters:
        self.vocabulary_ = []  # list of unique terms
        self.df_ = {}          # document frequency per term
        self.idf_ = {}         # inverse document frequency per term
        self.total_docs_ = 0   # total number of documents

        # Runtime tools (not pickled)
        self._stopwords = set(stopwords.words("english")) if remove_stopwords else set()
        self._lemmatizer = WordNetLemmatizer() if lemmatize else None

    # ----------------------------------------------------------------------
    # SAFE PICKLING
    # ----------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        # remove unpicklable objects
        if "_lemmatizer" in state:
            del state["_lemmatizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # restore tools
        if self.remove_stopwords:
            self._stopwords = set(stopwords.words("english"))
        else:
            self._stopwords = set()
        if self.lemmatize:
            self._lemmatizer = WordNetLemmatizer()
        else:
            self._lemmatizer = None

    # ----------------------------------------------------------------------
    # PREPROCESSING
    # ----------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text according to instance settings.
        """
        if text is None:
            return []

        if self.lowercase:
            text = text.lower()

        # (you can add punctuation removal here if you want)
        tokens = re.findall(r'\b\w+\b', text)

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        if self.lemmatize:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    @staticmethod
    def _count_words(tokens):
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return counts

    # ----------------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------------
    def _build_vocabulary(self, documents):
        vocab = set()
        for doc in documents:
            vocab.update(self._tokenize(doc or ""))
        self.vocabulary_ = sorted(vocab)

    def _compute_df(self, documents):
        df = {}
        for doc in documents:
            for t in set(self._tokenize(doc or "")):
                df[t] = df.get(t, 0) + 1
        self.df_ = df
        self.total_docs_ = len(documents)

    def _compute_idf(self):
        N = self.total_docs_
        self.idf_ = {
            w: float(np.log((N + 1) / (self.df_.get(w, 0) + 1)))
            for w in self.vocabulary_
        }

    def fit(self, documents):
        """
        Fit the TF-IDF model on a collection of documents.

        Parameters
        ----------
        documents : list of str
            A list where each element is the **content** of a document
            (raw text), NOT a file path or filename.
            Preprocessing (lowercasing, stopword removal, lemmatization)
            is applied internally according to the settings provided
            in __init__().

        Returns
        -------
        self
            The fitted TFIDF instance.
        """
        self._build_vocabulary(documents)
        self._compute_df(documents)
        self._compute_idf()
        return self

    # ----------------------------------------------------------------------
    # TRANSFORM  (ALWAYS RETURNS FULL MATRIX)
    # ----------------------------------------------------------------------
    def transform(self, documents):
        """
        Transform documents into TF-IDF representation using the vocabulary
        and IDF statistics learned during fit().

        Parameters
        ----------
        documents : list of str
            A list of documents to transform. Each element is the **raw text**
            of a document (not a path). The same preprocessing pipeline
            (lowercase, stopword removal, lemmatization) is applied as in fit().

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (n_documents, n_vocabulary) with TF-IDF scores.
        """
        vocab = self.vocabulary_
        V = len(vocab)
        D = len(documents)
        matrix = np.zeros((D, V))

        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            total = len(tokens)
            if total == 0:
                continue

            counts = self._count_words(tokens)

            for j, word in enumerate(vocab):
                tf = counts.get(word, 0) / total
                idf = self.idf_.get(word, 0)
                matrix[i, j] = tf * idf

        return matrix

    def fit_transform(self, documents):
        """
        Fit the model and transform documents in one step.
        """
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self):
        """Return the vocabulary list."""
        return self.vocabulary_

    # ----------------------------------------------------------------------
    # TOP-N KEYWORDS HELPER (USES self.top_n BY DEFAULT)
    # ----------------------------------------------------------------------
    def get_top_keywords(self, matrix: np.ndarray, top_n: int | None = None):
        """
        Compute top-N keywords per document from a TF-IDF matrix.

        Parameters
        ----------
        matrix : numpy.ndarray
            TF-IDF matrix of shape (n_documents, n_vocabulary).
        top_n : int or None, optional
            Number of top keywords per document. If None, uses self.top_n.

        Returns
        -------
        list[list[tuple[str, float]]]
            For each document, a list of (word, score) pairs, sorted by score descending.
        """
        if top_n is None:
            top_n = self.top_n

        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")

        vocab = self.vocabulary_
        top_keywords_per_doc: list[list[tuple[str, float]]] = []

        for row in matrix:
            if row.sum() == 0:
                top_keywords_per_doc.append([])
                continue

            idx = row.argsort()[::-1][:top_n]
            doc_keywords = [
                (vocab[j], row[j]) for j in idx if row[j] > 0
            ]
            top_keywords_per_doc.append(doc_keywords)

        return top_keywords_per_doc

    # ----------------------------------------------------------------------
    # SAVE / LOAD
    # ----------------------------------------------------------------------
    def save_to_file(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
