import numpy as np

from rag.embeddings import create_embedding


# ==================================================
# Retrieve Brochure Chunks
# ==================================================

def retrieve(
    query,
    index,
    chunks,
    brand,
    car_model,
    top_k=5,
    search_k=100
):

    # ----------------------------------------------
    # Create query embedding
    # ----------------------------------------------

    query_embedding = create_embedding(
        query,
        task_type="RETRIEVAL_QUERY"
    )

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )


    # ----------------------------------------------
    # Search FAISS
    # ----------------------------------------------

    k = min(
        search_k,
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []


    # ----------------------------------------------
    # Filter by selected vehicle
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

        # Only keep chunks belonging to
        # the selected vehicle
        if (
            chunk_brand.lower() == brand.lower()
            and
            chunk_model.lower() == car_model.lower()
        ):

            results.append({
                "score": float(score),
                "text": chunk.get(
                    "text",
                    ""
                ),
                "metadata": metadata
            })

        # Stop once enough vehicle-specific
        # results have been found
        if len(results) >= top_k:
            break


    return results


# ==================================================
# Local Test
# ==================================================

if __name__ == "__main__":

    from rag.vector_store import load_vector_store

    index, chunks = load_vector_store()

    query = "What safety features does the Sierra have?"

    results = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        brand="Tata",
        car_model="Sierra",
        top_k=5
    )

    print("\nQuestion:")
    print(query)

    print(
        "\nResults found:",
        len(results)
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "=" * 60)

        print(
            "RESULT",
            number
        )

        print(
            "Score:",
            round(
                result["score"],
                4
            )
        )

        print(
            "Metadata:",
            result["metadata"]
        )

        print("\nText:")
        print(result["text"])