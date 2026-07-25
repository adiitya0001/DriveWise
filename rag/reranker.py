from sentence_transformers import CrossEncoder


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def rerank(query, results, top_k=3):

    if not results:
        return []

    pairs = [
        [query, result["text"]]
        for result in results
    ]

    scores = reranker.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    ranked_results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked_results[:top_k]

if __name__ == "__main__":

    from ingestion.pdf_loader import load_pdf
    from ingestion.chunker import create_chunks
    from rag.retriever import retrieve

    pages = load_pdf(
        "data/brochures/creta.pdf",
        brand="Hyundai",
        model="Creta"
    )

    chunks = create_chunks(pages)

    query = "How many airbags does Creta have?"

    # First-stage retrieval
    results = retrieve(
        query=query,
        chunks=chunks,
        brand="Hyundai",
        car_model="Creta",
        top_k=5
    )

    # Second-stage reranking
    ranked_results = rerank(
        query=query,
        results=results,
        top_k=3
    )

    print("\nQuestion:", query)

    for number, result in enumerate(ranked_results, start=1):

        print("\n" + "=" * 60)
        print("RESULT", number)
        print("FAISS Score:", round(result["score"], 4))
        print(
            "Reranker Score:",
            round(result["rerank_score"], 4)
        )
        print("Metadata:", result["metadata"])

        print("\nText:")
        print(result["text"][:500])