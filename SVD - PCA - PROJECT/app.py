import streamlit as st
import numpy as np
import pickle
from tfidf import TFIDF
import plotly.express as px

# -----------------------------
# 1. APP TITLE
# -----------------------------
st.title("TF-IDF Document Processor")  # set the title of the app
st.markdown(
    "Upload text documents, process them with TF-IDF, analyze top keywords, "
    "visualize document distribution, and document your observations."
)  # description of the app

# -----------------------------
# 2. FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload .txt files",
    type=["txt"],
    accept_multiple_files=True
)  # allow multiple file uploads
if uploaded_files:
    st.write(f"{len(uploaded_files)} files uploaded.")  # display number of uploaded files
filenames = [f.name for f in uploaded_files] if uploaded_files else []  # get list of filenames

# -----------------------------
# 3. PREPROCESSING OPTIONS
# -----------------------------
st.header("Preprocessing Options")  # header for preprocessing options
to_lower = st.checkbox("Convert to lowercase", value=True)  # checkbox for lowercase conversion
remove_sw = st.checkbox("Remove stopwords", value=False)  # checkbox for stopword removal
do_lemma = st.checkbox("Lemmatize", value=False)  # checkbox for lemmatization

# display selected options
st.write(f"Selected preprocessing options: "
         f"lowercase={to_lower}, remove_stopwords={remove_sw}, lemmatize={do_lemma}"
)  # display selected options

# -----------------------------
# 4. DIMENSIONALITY REDUCTION OPTIONS
# -----------------------------
st.header("Dimensionality Reduction Options")  # header for dimensionality reduction options
method = st.selectbox("Choose reduction method:", ["PCA", "SVD"])  # selectbox for reduction method
dim = st.radio("Choose number of dimensions:", [2, 3])  # radio buttons for number of dimensions

# -----------------------------
# 5. PROCESS DOCUMENTS
# -----------------------------
if st.button("Process Documents"):  # button to process documents

    if not uploaded_files:  # check if any files were uploaded
        st.error("Please upload at least one .txt file.")  # error message if no files uploaded
        st.stop()  # stop execution if no files

    # --- Read raw text ---
    raw_docs = [f.read().decode("utf-8") for f in uploaded_files]  # read the content of uploaded files

    # --- Fit TFIDF ---
    model = TFIDF(lowercase=to_lower, remove_stopwords=remove_sw, lemmatize=do_lemma)  # create TFIDF model instance
    model.fit(raw_docs)  # fit the model to the raw documents
    full_vectors = model.transform(raw_docs) # transform documents to TF-IDF vectors without limiting to top N keywords
    vocab_list = model.get_feature_names()  # get the vocabulary list from the model

    # --- Top-N keywords per document ---
    top_keywords_per_doc = model.get_top_keywords(full_vectors) # uses self.top_n by default
    
    # --- Union keywords & filtered matrix ---
    union_keywords = sorted(
        set(word for doc in top_keywords_per_doc for word, _ in doc))  # get union of all top keywords
    filtered_vectors = np.zeros((len(raw_docs), len(union_keywords)))  # initialize filtered TF-IDF matrix
    for i, doc_keywords in enumerate(top_keywords_per_doc):  # iterate documents
        for word, score in doc_keywords:  # iterate top keywords
            j = union_keywords.index(word)  # find index in union keywords
            filtered_vectors[i, j] = score  # set score in filtered matrix

    # --- Store in session_state ---
    st.session_state.model = model  # store TFIDF model
    st.session_state.filtered_vectors = filtered_vectors  # store filtered TF-IDF matrix
    st.session_state.union_keywords = union_keywords  # store union of keywords
    st.session_state.filenames = filenames  # store filenames
    st.session_state.top_keywords_per_doc = top_keywords_per_doc  # store top keywords per document
    st.session_state.doc_analysis = {fname: "" for fname in filenames}  # initialize per-document analysis
    st.session_state.general_analysis = ""  # initialize general analysis

    st.success("Documents processed successfully!")  # success message
    st.write("Sample filtered keywords:", union_keywords[:20])  # display sample of filtered keywords
    st.write("Filtered matrix shape:", filtered_vectors.shape)  # display shape of filtered matrix

    # --- Save TFIDF model ---
    model_bytes = pickle.dumps(model)  # serialize the TFIDF model
    st.download_button("Save TFIDF Model", model_bytes, file_name="tfidf_model.pkl")  # download button for the model

# -----------------------------
# 6+7. VISUALIZATION & ANALYSIS
# -----------------------------
if "filtered_vectors" in st.session_state:  # check if documents have been processed
    
    model = st.session_state.model  # retrieve TFIDF model
    filtered_vectors = st.session_state.filtered_vectors  # retrieve filtered TF-IDF matrix
    filenames = st.session_state.filenames  # retrieve filenames
    top_keywords_per_doc = st.session_state.top_keywords_per_doc  # retrieve top keywords per document

    # --- Dropdown to select document ---
    selected_doc = st.selectbox("Select a document to inspect:",
                                ["(None)"] + filenames)  # dropdown for document selection
    selected_index = None  # initialize selected index
    if selected_doc != "(None)":  # if a document is selected
        selected_index = filenames.index(selected_doc)  # get index of selected document

    X = filtered_vectors  # TF-IDF matrix for dimensionality reduction

    # --- Dimensionality Reduction ---
    if method == "PCA":
        X_centered = X - X.mean(axis=0)  # center the data
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # perform SVD on centered data
    else:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)  # perform SVD on original data

    # Handle edge cases: empty matrix or fewer components than requested
    n_components = min(dim, U.shape[1]) if U.shape[1] > 0 else 0

    if n_components == 0:
        # No valid components - create zero matrix
        reduced = np.zeros((len(filenames), dim))
    else:
        U = U[:, :n_components]  # top components
        S = S[:n_components]     # keep singular values

        # Fix the sign of each component to be consistent across machines
        for k in range(n_components):
            if U[0, k] < 0:  # use first row as reference
                U[:, k] *= -1

        reduced = U * S  # shape: (n_docs, n_components)

        # Pad with zeros if we got fewer components than requested
        if n_components < dim:
            reduced = np.hstack([reduced, np.zeros((len(filenames), dim - n_components))])

    # --- Prepare Plotly Data ---
    if dim == 2:  # 2D plot
        colors = ["red" if i == selected_index else "blue" for i in range(len(filenames))]  # color coding
        sizes = [15 if i == selected_index else 8 for i in range(len(filenames))]  # size coding
        fig = px.scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            hover_name=filenames
        )  # create 2D scatter plot
        fig.update_traces(marker=dict(size=sizes, color=colors))  # update marker properties
        fig.update_layout(xaxis_title="Component 1", yaxis_title="Component 2")  # set axis titles
    else:  # 3D plot
        colors = ["red" if i == selected_index else "blue" for i in range(len(filenames))]  # color coding
        sizes = [8 if i != selected_index else 15 for i in range(len(filenames))]  # size coding
        fig = px.scatter_3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            hover_name=filenames
        )  # create 3D scatter plot
        fig.update_traces(marker=dict(size=sizes, color=colors))  # update marker properties
        fig.update_layout(
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3"
            )
        )  # set axis titles

    st.plotly_chart(fig, use_container_width=True)  # display the plotly chart

    # --- Top Keywords Table (per document) ---
    if selected_index is not None:  # if a document is selected
        num_keywords_available = len(top_keywords_per_doc[selected_index])
        effective_top_n = min(model.top_n, num_keywords_available)

        st.subheader(
            f"Top {effective_top_n} Keywords for {selected_doc}"
        )
        keywords_table = top_keywords_per_doc[selected_index]  # get top keywords for selected document
        if keywords_table:  # if there are keywords
            df_kw = {
                "Keyword": [kw for kw, score in keywords_table],
                "TF-IDF Score": [score for kw, score in keywords_table]
            }  # prepare data for dataframe
            st.dataframe(df_kw)  # display the dataframe
        else:  # if no keywords
            st.write("No keywords available for this document.")

    # -----------------------------
    # 6. General Analysis (always visible)
    # -----------------------------
    st.header("General Analysis")  # header for general analysis
    st.markdown(""" 
    **Overall Observations:**  
    Use this section to document your general observations across all documents, regardless of which specific document is selected.

    Questions to guide your analysis:
    - What patterns do you observe in the document distribution?
    - Are there clusters of similar documents? What makes them similar?
    - What do the distances between documents tell you?
    - Do the top keywords help explain the clustering?
    - How do different preprocessing options affect the clustering?
    - How does PCA vs SVD affect the visualization?
    - Any other interesting observations?
    """)
    general_analysis_text = st.text_area(
        "Write your overall observations here:",
        value=st.session_state.general_analysis,
        height=200
    )  # text area for general analysis
    st.session_state.general_analysis = general_analysis_text  # store general analysis

    # -----------------------------
    # 7. Per-document Analysis
    # -----------------------------
    if selected_index is not None:  # if a document is selected
        st.header(f"Analysis for '{selected_doc}'")  # header for per-document analysis
        doc_analysis_text = st.text_area(
            f"Write your observations for '{selected_doc}':",
            value=st.session_state.doc_analysis[selected_doc],
            height=200
        )  # text area for per-document analysis
        st.session_state.doc_analysis[selected_doc] = doc_analysis_text  # store per-document analysis