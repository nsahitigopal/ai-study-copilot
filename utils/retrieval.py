from utils.embeddings import (
    embed_query
)

from utils.vector_store import (
    search_faiss_index
)


def retrieve_context(
    query,
    index,
    chunks,
    k=5
):

    query_embedding = (
        embed_query(query)
    )

    results = (
        search_faiss_index(
            index,
            query_embedding,
            chunks,
            k
        )
    )

    context = "\n\n".join(

        result["chunk"].page_content

        for result in results
    )

    return context