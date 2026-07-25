import faiss
import numpy as np

from rag.embeddings import model



def retrieve(
    query,
    index,
    chunks,
    brand,
    car_model,
    top_k=5,
    search_k=20
):
    # Embed ONLY the user's question
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Search more candidates than we ultimately need
    k = min(search_k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = chunks[idx]
        metadata = chunk["metadata"]

        # Metadata filtering
        if (
            metadata["brand"].lower() == brand.lower()
            and metadata["model"].lower() == car_model.lower()
        ):
            results.append({
                "score": float(score),
                "text": chunk["text"],
                "metadata": metadata
            })

        # Stop once we have enough valid results
        if len(results) >= top_k:
            break

    return results

if __name__ == "__main__":

    from rag.vector_store import load_vector_store

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

    print("\nQuestion:", query)

    for number, result in enumerate(results, start=1):

        print("\n" + "=" * 60)
        print("RESULT", number)
        print("Score:", round(result["score"], 4))
        print("Metadata:", result["metadata"])

        print("\nText:")
        print(result["text"][:500])