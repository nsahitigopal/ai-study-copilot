import streamlit as st
import numpy as np
import faiss

from utils.pdf_processing import extract_text_from_pdf
from utils.preprocessing import clean_text
from utils.chunking import chunk_text
from utils.embeddings import generate_embeddings

st.title("AI Study Copilot")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file:

    # -----------------------------
    # EXTRACT TEXT
    # -----------------------------
    extracted_text = extract_text_from_pdf(
        uploaded_file
    )

    st.subheader("Extracted Text")

    st.write(extracted_text[:5000])

    # -----------------------------
    # CLEAN TEXT
    # -----------------------------
    cleaned_text = clean_text(
        extracted_text
    )

    st.subheader("Cleaned Text")

    st.write(cleaned_text[:5000])

    # -----------------------------
    # CHUNK TEXT
    # -----------------------------
    chunks = chunk_text(cleaned_text)

    st.subheader("Generated Chunks")

    for i, chunk in enumerate(chunks[:5]):

        st.write(f"Chunk {i+1}")

        st.write(chunk)

        st.write("------")

    # -----------------------------
    # GENERATE EMBEDDINGS
    # -----------------------------
    st.subheader("Generating Embeddings...")

    embeddings = generate_embeddings(chunks)

    st.write("Embedding Shape:")

    st.write(embeddings.shape)

    # -----------------------------
    # NORMALIZE EMBEDDINGS
    # FOR COSINE SIMILARITY
    # -----------------------------
    faiss.normalize_L2(embeddings)

    # -----------------------------
    # CREATE FAISS INDEX
    # -----------------------------
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    st.success(
        "Embeddings stored in FAISS!"
    )

    # -----------------------------
    # USER QUESTION
    # -----------------------------
    question = st.text_input(
        "Ask a question from the PDF"
    )

    if question:

        # -----------------------------
        # EMBED QUESTION
        # -----------------------------
        question_embedding = generate_embeddings(
            [question]
        )

        faiss.normalize_L2(
            question_embedding
        )

        # -----------------------------
        # SEARCH FAISS
        # -----------------------------
        k = 3

        distances, indices = index.search(
            question_embedding,
            k
        )

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        st.subheader(
            "Most Relevant Chunks"
        )

        for i, idx in enumerate(
            indices[0]
        ):

            st.write(
                f"### Result {i+1}"
            )

            st.write(chunks[idx])

            st.write("---")