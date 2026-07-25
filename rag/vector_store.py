import os
import pickle

import faiss

from rag.embeddings import create_embeddings


# ==================================================
# Paths
# ==================================================

VECTOR_STORE_DIR = "data/vector_store"

INDEX_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "index.faiss"
)

CHUNKS_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "chunks.pkl"
)


# ==================================================
# Build Vector Store
# ==================================================

def build_vector_store(chunks):

    if not chunks:
        raise ValueError(
            "Cannot build vector store with no chunks."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = create_embeddings(texts)

    dimension = embeddings.shape[1]

    print(
        "Embedding dimension:",
        dimension
    )

    # Inner product + normalized vectors
    # behaves like cosine similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index, chunks


# ==================================================
# Save Vector Store
# ==================================================

def save_vector_store(index, chunks):

    os.makedirs(
        VECTOR_STORE_DIR,
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(
        CHUNKS_PATH,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print(
        "Vector store saved successfully."
    )


# ==================================================
# Load Vector Store
# ==================================================

def load_vector_store():

    if not os.path.exists(INDEX_PATH):

        raise FileNotFoundError(
            "FAISS index not found. "
            "Build the vector store first."
        )

    if not os.path.exists(CHUNKS_PATH):

        raise FileNotFoundError(
            "Chunks file not found. "
            "Build the vector store first."
        )

    index = faiss.read_index(
        INDEX_PATH
    )

    with open(
        CHUNKS_PATH,
        "rb"
    ) as file:

        chunks = pickle.load(file)

    return index, chunks