from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# LOAD MODEL
# -----------------------------
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------
def generate_embeddings(text_chunks):

    embeddings = model.encode(text_chunks)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    return embeddings