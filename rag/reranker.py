def rerank(query, results, top_k=5):
    """
    Lightweight production reranker.

    FAISS already ranks retrieved chunks using semantic
    similarity. We keep that ranking and return the
    top results.

    Keeping 5 chunks gives the answer-generation model
    more brochure context, especially for broad questions
    such as safety features, ADAS, comfort, etc.

    This avoids loading a second transformer model,
    reducing RAM usage in production.
    """

    if not results:
        return []

    return results[:top_k]


if __name__ == "__main__":
    print("DriveWise lightweight reranker is active.")