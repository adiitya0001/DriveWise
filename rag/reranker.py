def rerank(query, results, top_k=3):
    """
    Lightweight production reranker.

    FAISS already ranks retrieved chunks using semantic
    similarity. We keep that ranking and simply return
    the top results.

    This avoids loading a second transformer model,
    reducing RAM usage in production.
    """

    if not results:
        return []

    return results[:top_k]


if __name__ == "__main__":
    print("DriveWise lightweight reranker is active.")