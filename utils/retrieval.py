from utils.embeddings import embed_query
from utils.vector_store import search_faiss_index


def retrieve_context(query, index, chunks, k=5):
    """
    What enters : user query (str), FAISS index, list of Document chunks, k
    What leaves : dict with:
                    "context"      — top-k chunk texts joined as one string
                    "source_pages" — sorted 1-indexed page numbers
    Why         : connects the question to relevant parts of the document.
                  The LLM receives the context string.
                  The UI shows source pages so the student can verify.
    """

    # Embed the query using the same model used for the chunks
    # Same model = same vector space = meaningful similarity comparison
    query_embedding = embed_query(query)

    results = search_faiss_index(index, query_embedding, chunks, k)


    # Join the top-k chunk texts into one block of context
    context = "\n\n".join(
        result["chunk"].page_content
        for result in results
    )


    # chunk metadata["page"] is 0-indexed (page 0 = page 1 of the PDF)
    # Add 1 to convert to reader-friendly page numbers
    source_pages = sorted(set(
        result["chunk"].metadata.get("page", 0) + 1
        for result in results
    ))

    return {
        "context":      context,
        "source_pages": source_pages,
    }
