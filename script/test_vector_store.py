from rag.vector_store import load_vector_store


index, chunks = load_vector_store()

print("FAISS vectors:", index.ntotal)
print("Stored chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0])