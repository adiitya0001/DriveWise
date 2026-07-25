import os
import pickle

import faiss

from rag.embeddings import model


VECTOR_STORE_DIR = "data/vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_STORE_DIR, "chunks.pkl")


def build_vector_store(chunks):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, chunks


def save_vector_store(index, chunks):

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    # Save chunks + metadata
    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)

    print("Vector store saved successfully.")


def load_vector_store():

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "FAISS index not found. Build the vector store first."
        )

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "Chunks file not found. Build the vector store first."
        )

    # Load FAISS
    index = faiss.read_index(INDEX_PATH)

    # Load chunks
    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    return index, chunks