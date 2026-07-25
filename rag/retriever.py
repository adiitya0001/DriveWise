import faiss
import numpy as np

from rag.embeddings import model


# ==================================================
# Retrieve Relevant Brochure Chunks
# ==================================================

def retrieve(
    query,
    index,
    chunks,
    brand,
    car_model,
    top_k=5,
    search_k=20
):
    """
    Retrieve brochure chunks relevant to the user's
    question while filtering by selected brand/model.
    """

    # ----------------------------------------------
    # 1. Create query embedding
    # ----------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # ----------------------------------------------
    # 2. Search FAISS
    # ----------------------------------------------

    k = min(search_k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    # ----------------------------------------------
    # 3. Filter results by vehicle
    # ----------------------------------------------

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        chunk = chunks[idx]

        metadata = chunk.get(
            "metadata",
            {}
        )

        chunk_brand = metadata.get(
            "brand",
            ""
        )

        chunk_model = metadata.get(
            "model",
            ""
        )

        if (
            chunk_brand.lower() == brand.lower()
            and
            chunk_model.lower() == car_model.lower()
        ):

            results.append({
                "score": float(score),
                "text": chunk.get("text", ""),
                "metadata": metadata
            })

        # We have enough matching chunks
        if len(results) >= top_k:
            break

    return results


# ==================================================
# Local Test
# ==================================================

if __name__ == "__main__":

    from rag.vector_store import load_vector_store

    print("Loading vector store...")

    index, chunks = load_vector_store()

    query = "How many airbags does Creta have?"

    results = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        brand="Hyundai",
        car_model="Creta",
        top_k=5
    )

    print("\nQuestion:")
    print(query)

    print(
        f"\nRetrieved {len(results)} results"
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "=" * 60)
        print("RESULT", number)

        print(
            "Score:",
            round(result["score"], 4)
        )

        print(
            "Metadata:",
            result["metadata"]
        )

        print("\nText:")
        print(
            result["text"][:500]
        )