import faiss
import numpy as np


def create_faiss_index(
    embeddings
):

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = (
        embeddings.shape[1]
    )

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


def search_faiss_index(
    index,
    query_embedding,
    chunks,
    k=5
):

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    distances, indices = (
        index.search(
            query_embedding,
            k
        )
    )

    results = []

    for rank, idx in enumerate(
        indices[0]
    ):

        results.append(
            {
                "chunk": chunks[idx],
                "distance": float(
                    distances[0][rank]
                )
            }
        )

    return results


def save_index(
    index,
    file_path
):

    faiss.write_index(
        index,
        file_path
    )


def load_index(
    file_path
):

    index = faiss.read_index(
        file_path
    )

    return index